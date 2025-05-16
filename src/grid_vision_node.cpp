#include "grid_vision/grid_vision_node.hpp"
#include "grid_vision/object_detection.hpp"

GridVision::GridVision() : Node("grid_vision_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  lidar_topic_ = declare_parameter<std::string>("lidar_topic", "");
  det_weight_file_ = declare_parameter<std::string>("detection_weights_file", "");
  depth_weight_file_ = declare_parameter<std::string>("depth_weights_file", "");
  lidar_frame_ = declare_parameter<std::string>("lidar_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  conf_threshold_ = declare_parameter("confidence_threshold", 0.5);
  iou_threshold_ = declare_parameter("iou_threshold", 0.4);
  resize_ = declare_parameter("detection_network_input_size", 416);
  depth_input_h_ = declare_parameter("depth_network_height", 192);
  depth_input_w_ = declare_parameter("depth_network_width", 640);
  cam_height_ = declare_parameter("camera_image_height", 480);
  cam_width_ = declare_parameter("camera_image_width", 640);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);
  k_near_ = declare_parameter("k_near", 10);
  patch_size_ = declare_parameter("patch_size", 5);
  grid_x_ = declare_parameter("grid_x", 50);
  grid_y_ = declare_parameter("grid_y", 10);
  resolution_ = declare_parameter("resolution", 0.1);
  camera_only_ = declare_parameter("camera_only", false);

  // Initialize occupancy grid
  occ_grid_ = OccupancyGridMap(base_frame_, grid_x_, grid_y_, resolution_);

  if(image_topic_.empty() || det_weight_file_.empty() || lidar_topic_.empty())
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
  depth_img_pub_ = image_transport::create_publisher(this, "carla/front/depth_image");
  occupancy_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 10);

  // Get weight paths
  std::string share_dir = ament_index_cpp::get_package_share_directory("grid_vision");
  std::string det_weight_path = share_dir + det_weight_file_;
  std::string depth_weight_path = share_dir + depth_weight_file_;

  // Initialize ONNX runtime
  object_detection::initialize_onnx_runtime(session_, env_, session_options_,
                                            det_weight_path.c_str());

  // Initialize TensorRT and depthEstimation class
  monodepth_ = MonoDepthEstimation(depth_input_h_, depth_input_w_, cam_height_,
                                   cam_width_, patch_size_, depth_weight_path);

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Set Intrinsic Matrix
  intrinsic_mat_ = object_detection::setIntrinsicMatrix(fx_, fy_, cx_, cy_);
  // Get Intrinsic Matrix Inverse
  K_inv_ = object_detection::computeKInverse(intrinsic_mat_);
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
  if(init_image_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image is missing in GridVision");
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  if(cloud_.empty() && !camera_only_)
  {
    RCLCPP_WARN(this->get_logger(), "PointCloud is missing in GridVision");
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }
  // Preprocess the image
  // Convert it to tensors
  cv::Mat preprocessed
    = object_detection::preprocess_image(init_image_, resize_, resize_);
  std::vector<float> input_tensor = object_detection::mat_to_tensor(preprocessed);

  // Perform inference
  // std::vector<Ort::Value> output
  //  = object_detection::run_inference(input_tensor, session_);

  // Extract Bboxes
  // std::vector<BoundingBox> bboxes = object_detection::extract_bboxes(
  //  output, conf_threshold_, iou_threshold_, init_image_.cols, init_image_.rows, resize_);

  // If bboxes are empty then nothing to do
  // if(bboxes.empty())
  //{
  // Update the map
  //  occ_grid_->updateMap(occ_grid_->grid_map_);
  //  publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
  //  return;
  //}
  BoundingBox bbox;
  bbox.confidence = 0.2;
  bbox.x_max = 200;
  bbox.y_max = 50;
  bbox.x_min = 150;
  bbox.y_min = 25;
  std::vector<BoundingBox> bboxes;
  bboxes.emplace_back(bbox);
  if(!camera_only_) // use LIDAR
  {
    // Transform point cloud to camera frame
    // return if not transformed_cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud
      = transformLidarToCamera(cloud_, lidar_frame_, camera_frame_);

    if(!transformed_cloud)
    {
      publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
      return;
    }

    // Build KDTree for storing transformed point (3d to 2d pixel)
    pcl::PointCloud<pcl::PointXYZ>::Ptr image_points(
      new pcl::PointCloud<pcl::PointXYZ>());
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    cloud_detections::buildKDTree(kdtree, image_points, transformed_cloud,
                                  intrinsic_mat_);

    // Get depth of 2D detections from kdtree
    // depth size = size of bboxes
    depth_vec_ = cloud_detections::computeDepthForBoundingBoxes(kdtree, image_points,
                                                                bboxes, k_near_);
  }
  else // use monocular depth estimation
  {
    RCLCPP_ERROR(this->get_logger(), "RUN THE DEPTH ESTIMATION");
    cv::Mat init_clone = init_image_.clone();
    depth_vec_ = monodepth_->runInference(init_clone, bboxes);
    RCLCPP_ERROR(this->get_logger(), "DEPTH ESTIMATION DONE");
  }

  // Get the 3D co-ordinates of 2D detection in base frame
  // std::vector<geometry_msgs::msg::Point> cam_points
  //  = convertPixelsTo3D(bboxes, depth_vec_, K_inv_);
  // Update the occupancy grid map
  // occ_grid_->updateMap(occ_grid_->grid_map_, cam_points, bboxes);

  // Publish 2D detections and Occupancy grid
  // publishObjectDetections(detection_pub_, bboxes, init_image_, resize_);
  publishDepthImage(depth_img_pub_);
  // publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
  RCLCPP_ERROR(this->get_logger(), "LOOP COMPLETE");
}

void GridVision::publishObjectDetections(const image_transport::Publisher &pub,
                                         std::vector<BoundingBox> &bboxes, cv::Mat &image,
                                         int resize)
{
  cv::Mat bbox_img = image.clone();

  // Draw Bboxes
  object_detection::draw_bboxes(bbox_img, bboxes);

  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, "rgb8", bbox_img).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

void GridVision::publishDepthImage(const image_transport::Publisher &pub)
{
  cv::Mat bbox_img = monodepth_->depth_img_;
  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, "mono8", bbox_img).toImageMsg();

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
                              const std::vector<float> &depths,
                              const Eigen::Matrix3d &K_inv)
{
  std::vector<geometry_msgs::msg::Point> point_vec;

  for(size_t i = 0; i < bboxes.size(); ++i)
  {
    geometry_msgs::msg::Point cam_point;
    // Compute the pixel center
    cv::Point2f pixel_center(
      bboxes[i].x_min + ((bboxes[i].x_max - bboxes[i].x_min) / 2.0f),
      bboxes[i].y_min + ((bboxes[i].y_max - bboxes[i].y_min) / 2.0f));

    if(!camera_only_)
    {
      // Convert to 3D point in camera frame
      cam_point = cloud_detections::pixelTo3D(pixel_center, depths[i], K_inv);
    }
    else
    {
      cam_point = monodepth_->pixelTo3D(pixel_center, depths[i], K_inv);
    }

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
