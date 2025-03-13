#include "grid_vision/grid_vision_node.hpp"
#include "grid_vision/cloud_detections.hpp"

GridVision::GridVision() : Node("grid_vision_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  weight_file_ = declare_parameter<std::string>("weights_file", "");
  lidar_frame_ = declare_parameter<std::string>("lidar_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  conf_threshold_ = declare_parameter("confidence_threshold", 0.5);
  iou_threshold_ = declare_parameter("iou_threshold", 0.4);
  resize_ = declare_parameter("network_input_size", 416);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0);
  cy_ = declare_parameter("cy", 0);
  k_near_ = declare_parameter("k_near", 10);

  if(image_topic_.empty() || weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Check if topic name or weight file is assigned");
    return;
  }

  // Image Transport for subscribing
  image_transport::ImageTransport it(shared_from_this());
  image_sub_ = it.subscribe(
    image_topic_, 1, std::bind(&GridVision::imageCallback, this, std::placeholders::_1));
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/camera/pointcloud", 10,
    std::bind(&GridVision::cloudCallback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&GridVision::timerCallback, this));

  detection_pub_ = it.advertise("carla/rgb/front/detections", 1);

  // Initialize ONNX runtim
  session_ = object_detection::initialize_onnx_runtime(env_, session_options_,
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
    init_image_ = cv_bridge::toCvCopy(msg, "rgb8")->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
}

void GridVision::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  pcl::fromROSMsg(*msg, *cloud_);
}

void GridVision::timerCallback()
{
  // Check if teh image nd pointcloud exists
  if(init_image_.empty() || cloud_->empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image or Pointcloud is missing in GridVision");
    return;
  }

  // Transform point cloud to camera frame
  // return if not transformed_cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud
    = transformLidarToCamera(cloud_, lidar_frame_, camera_frame_);

  if(transformed_cloud->empty())
    return;

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
    = object_detection::extract_bboxes(output[0], conf_threshold_);

  // If bboxes are empty then nothing to do
  if(bboxes.empty())
    return;

  // Build KDTree for storing transformed point (3d to 2d pixel)
  pcl::PointCloud<pcl::PointXYZ>::Ptr image_points(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  cloud_detections::buildKDTree(kdtree, image_points, transformed_cloud, intrinsic_mat_);

  // Get depth of 2D detections from kdtree
  // depth size = size of bboxes
  std::vector<float> depth = cloud_detections::computeDepthForBoundingBoxes(
    kdtree, image_points, bboxes, k_near_);

  // Publish 2D detections
  publishObjectDetections(detection_pub_, bboxes, init_image_);
}

void GridVision::publishObjectDetections(const image_transport::Publisher &pub,
                                         std::vector<BoundingBox> &bboxes, cv::Mat &image)
{
  // Draw Bboxes
  object_detection::draw_bboxes(image, bboxes);

  // Convert OpenCV image to ROS2 message
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  sensor_msgs::msg::Image::SharedPtr msg
    = cv_bridge::CvImage(header, "rgb8", image).toImageMsg();

  // Publish image
  pub.publish(*msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
GridVision::transformLidarToCamera(const pcl::PointCloud<pcl::PointXYZ>::Ptr &lidar_cloud,
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
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(
    new pcl::PointCloud<pcl::PointXYZ>);
  tf2::Transform tf_transform;
  tf2::fromMsg(transform_stamped.transform, tf_transform);
  pcl_ros::transformPointCloud(*lidar_cloud, *transformed_cloud, tf_transform);

  return transformed_cloud;
}

int main(int argc, char *argv[])
{
  auto node = std::make_shared<GridVision>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
