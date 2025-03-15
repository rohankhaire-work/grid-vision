#include "grid_vision/grid_vision_node.hpp"
#include <image_transport/image_transport.hpp>
#include <pcl/impl/point_types.hpp>

GridVision::GridVision() : Node("grid_vision_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  lidar_topic_ = declare_parameter<std::string>("lidar_topic", "");
  weight_file_ = declare_parameter<std::string>("weights_file", "");
  lidar_frame_ = declare_parameter<std::string>("lidar_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  conf_threshold_ = declare_parameter("confidence_threshold", 0.5);
  iou_threshold_ = declare_parameter("iou_threshold", 0.4);
  resize_ = declare_parameter("network_input_size", 416);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);
  k_near_ = declare_parameter("k_near", 10);
  grid_x_ = declare_parameter("grid_x", 50);
  grid_y_ = declare_parameter("grid_y", 10);
  resolution_ = declare_parameter("resolution", 0.1);

  // Initialize occupancy grid
  occ_grid_ = OccupancyGridMap(base_frame_, grid_x_, grid_y_, resolution_);

  if(image_topic_.empty() || weight_file_.empty() || lidar_topic_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Check if topic name or weight file is assigned");
    return;
  }
  // Image Transport for subscribing
  image_sub_ = image_transport::create_subscription(
    this, image_topic_,
    std::bind(&GridVision::imageCallback, this, std::placeholders::_1), "raw");
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    lidar_topic_, 1, std::bind(&GridVision::cloudCallback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&GridVision::timerCallback, this));

  detection_pub_ = image_transport::create_publisher(this, "carla/front/detections");
  occupancy_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 10);

  // Initialize ONNX runtim
  object_detection::initialize_onnx_runtime(session_, env_, session_options_,
                                            weight_file_.c_str());

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Set Intrinsic Matrix
  intrinsic_mat_ = object_detection::setIntrinsicMatrix(fx_, fy_, cx_, cy_);
}

void GridVision::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  // Convert ROS2 image message to OpenCV format
  try
  {
    init_image_ptr_ = cv_bridge::toCvCopy(msg, "rgb8");

    // Check if the ptr is present
    if(!init_image_ptr_)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge::toCvCopy() returned nullptr!");
      return;
    }

    // Copy the image
    init_image_ = init_image_ptr_->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
}

void GridVision::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  pcl::fromROSMsg(*msg, cloud_);
}

void GridVision::timerCallback()
{
  // Check if the image and pointcloud exists
  if(init_image_.empty() || cloud_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image or Pointcloud is missing in GridVision");
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Transform point cloud to camera frame
  // return if not transformed_cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud
    = transformLidarToCamera(cloud_, lidar_frame_, camera_frame_);

  if(!transformed_cloud)
  {
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Preprocess the image
  // Convert it to tensors
  cv::Mat preprocessed
    = object_detection::preprocess_image(init_image_, resize_, resize_);
  std::vector<float> input_tensor = object_detection::mat_to_tensor(preprocessed);

  // Perform inference
  std::vector<Ort::Value> output
    = object_detection::run_inference(input_tensor, session_);

  // Extract Bboxes
  std::vector<BoundingBox> bboxes
    = object_detection::extract_bboxes(output, conf_threshold_, iou_threshold_, resize_);

  RCLCPP_INFO(this->get_logger(), "EXTRACTED BBOXES: %li", bboxes.size());

  // If bboxes are empty then nothing to do
  if(bboxes.empty())
  {
    // Update the map
    occ_grid_->updateMap(occ_grid_->grid_map_);
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Build KDTree for storing transformed point (3d to 2d pixel)
  pcl::PointCloud<pcl::PointXYZ>::Ptr image_points(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  cloud_detections::buildKDTree(kdtree, image_points, transformed_cloud, intrinsic_mat_);

  // Get depth of 2D detections from kdtree
  // depth size = size of bboxes
  std::vector<float> depth = cloud_detections::computeDepthForBoundingBoxes(
    kdtree, image_points, bboxes, k_near_);

  // Kinverse
  cv::Mat K_inv = object_detection::computeKInverse(intrinsic_mat_);

  // Get the 3D co-ordinates of 2D detection in base frame
  std::vector<geometry_msgs::msg::Point> cam_points
    = convertPixelsTo3D(bboxes, depth, K_inv);

  // Update the occupancy grid map
  occ_grid_->updateMap(occ_grid_->grid_map_, cam_points, bboxes);

  // Publish 2D detections and Occupancy grid
  publishObjectDetections(detection_pub_, bboxes, init_image_, resize_);
  publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
}

void GridVision::publishObjectDetections(const image_transport::Publisher &pub,
                                         std::vector<BoundingBox> &bboxes, cv::Mat &image,
                                         int resize)
{
  cv::Mat bbox_resize;

  // Resize to model input size
  cv::resize(image, bbox_resize, cv::Size(resize, resize));

  // Draw Bboxes
  object_detection::draw_bboxes(bbox_resize, bboxes);

  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, "rgb8", bbox_resize).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

void GridVision::publishOccupancyGrid(const grid_map::GridMap &grid_map,
                                      const std::string &base)
{
  // Convert GridMap to OccupancyGrid
  nav_msgs::msg::OccupancyGrid occupancy_grid;
  grid_map::GridMapRosConverter::toOccupancyGrid(grid_map, "occupancy", 0.0, 1.0,
                                                 occupancy_grid);

  // Set header info
  occupancy_grid.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  occupancy_grid.header.frame_id = base; // Set appropriate frame

  occupancy_pub_->publish(occupancy_grid);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr
GridVision::transformLidarToCamera(const pcl::PointCloud<pcl::PointXYZI> &lidar_cloud,
                                   const std::string &lidar_frame,
                                   const std::string &camera_frame)
{
  // Lookup transform from LiDAR to Camera frame
  geometry_msgs::msg::TransformStamped transform_stamped;
  try
  {
    transform_stamped
      = tf_buffer_->lookupTransform(camera_frame, lidar_frame, tf2::TimePointZero);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Could not transform %s to %s: %s", lidar_frame.c_str(),
                 camera_frame.c_str(), ex.what());
    return nullptr;
  }

  // Apply transformation to point cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(
    new pcl::PointCloud<pcl::PointXYZI>);
  tf2::Transform tf_transform;
  tf2::fromMsg(transform_stamped.transform, tf_transform);
  pcl_ros::transformPointCloud(lidar_cloud, *transformed_cloud, tf_transform);

  return transformed_cloud;
}

std::vector<geometry_msgs::msg::Point>
GridVision::convertPixelsTo3D(const std::vector<BoundingBox> &bboxes,
                              const std::vector<float> &depths, const cv::Mat &K_inv)
{
  std::vector<geometry_msgs::msg::Point> point_vec;

  for(size_t i = 0; i < bboxes.size(); ++i)
  {
    // Compute the pixel center
    cv::Point2f pixel_center((bboxes[i].x_max - bboxes[i].x_min) / 2.0f,
                             (bboxes[i].y_max - bboxes[i].y_min) / 2.0f);

    // Convert to 3D point in camera frame
    geometry_msgs::msg::Point cam_point
      = cloud_detections::pixelTo3D(pixel_center, depths[i], K_inv);

    // Transform to base frame
    geometry_msgs::msg::Point base_point
      = transformToBaseFrame(cam_point, camera_frame_, base_frame_);

    point_vec.emplace_back(base_point);
  }

  return point_vec;
}

geometry_msgs::msg::Point
GridVision::transformToBaseFrame(const geometry_msgs::msg::Point &cam_point,
                                 const std::string &source, const std::string &target)
{
  geometry_msgs::msg::Point base_point;

  try
  {
    // Lookup transform from camera frame to base frame
    geometry_msgs::msg::TransformStamped transform_stamped
      = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);

    // Transform the point
    tf2::doTransform(cam_point, base_point, transform_stamped);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Transform failed: %s", ex.what());
  }

  return base_point;
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GridVision>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
