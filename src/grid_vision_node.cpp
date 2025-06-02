#include "grid_vision/grid_vision_node.hpp"
#include "grid_vision/object_detection.hpp"
#include <memory>

GridVision::GridVision() : Node("grid_vision_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  lidar_topic_ = declare_parameter<std::string>("lidar_topic", "");
  det_weight_file_ = declare_parameter<std::string>("detection_weights_file", "");
  vis_weight_file_ = declare_parameter<std::string>("vision_weight_file", "");
  lidar_frame_ = declare_parameter<std::string>("lidar_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  conf_threshold_ = declare_parameter("confidence_threshold", 0.5);
  iou_threshold_ = declare_parameter("iou_threshold", 0.4);
  resize_ = declare_parameter("detection_network_input_size", 416);
  k_near_ = declare_parameter("k_near", 10);
  grid_x_ = declare_parameter("grid_x", 50);
  grid_y_ = declare_parameter("grid_y", 10);
  resolution_ = declare_parameter("resolution", 0.1);
  use_vision_orientation_ = declare_parameter("use_vision_orientation", false);

  // Set CAMParams Struct
  cam_params_.orig_h = declare_parameter("camera_image_height", 480);
  cam_params_.orig_w = declare_parameter("camera_image_width", 640);
  cam_params_.network_h = declare_parameter("network_height", 224);
  cam_params_.network_w = declare_parameter("network_width", 224);
  cam_params_.fx = declare_parameter("fx", 0.0);
  cam_params_.fy = declare_parameter("fy", 0.0);
  cam_params_.cx = declare_parameter("cx", 0.0);
  cam_params_.cy = declare_parameter("cy", 0.0);

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
  occupancy_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 10);
  viz_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("objects_viz", 10);

  // Get weight paths
  std::string share_dir = ament_index_cpp::get_package_share_directory("grid_vision");
  std::string det_weight_path = share_dir + det_weight_file_;

  // Initialize ONNX runtime
  object_detection::initialize_onnx_runtime(session_, env_, session_options_,
                                            det_weight_path.c_str());

  // Initialize Vision orientation network
  vision_orient_ = std::make_unique<VisionOrientation>(cam_params_, vis_weight_file_);

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Set Intrinsic Matrix
  intrinsic_mat_ = object_detection::setIntrinsicMatrix(cam_params_.fx, cam_params_.fy,
                                                        cam_params_.cx, cam_params_.cy);
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
  if(init_image_.empty() && cloud_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image or Pointcloud is missing in GridVision");
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Preprocess the image
  // Convert it to tensors
  cv::Mat preprocessed
    = object_detection::preprocess_image(init_image_, resize_, resize_);
  std::vector<float> input_tensor = object_detection::mat_to_tensor(preprocessed);

  // log time
  auto start_time = std::chrono::steady_clock::now();
  // Perform inference
  std::vector<Ort::Value> output
    = object_detection::run_inference(input_tensor, session_);

  // Get run time
  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms
    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  // RCLCPP_INFO(this->get_logger(), "Inference took %ld ms", duration_ms);

  // Extract Bboxes
  std::vector<BoundingBox> bboxes = object_detection::extract_bboxes(
    output, conf_threshold_, iou_threshold_, init_image_.cols, init_image_.rows, resize_);

  // If bboxes are empty then nothing to do
  if(bboxes.empty())
  {
    // Update the map
    occ_grid_->updateMap(occ_grid_->grid_map_);
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Filter bboxes vector for dynamic labels & static labels
  // dynamic - vehicle, person, bike & motorbike
  // static - traffic light, traffic sign
  auto [static_bboxes, dynamic_bboxes] = filterBBoxes(bboxes);

  // Transform point cloud to camera frame
  // return if not transformed_cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud
    = transformLidarToCamera(cloud_, lidar_frame_, camera_frame_);

  if(!transformed_cloud)
  {
    publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);
    return;
  }

  // Handle static bboxes
  std::vector<geometry_msgs::msg::Point> cam_points;
  if(!static_bboxes.empty())
  {
    // Build KDTree for storing transformed point (3d to 2d pixel)
    pcl::PointCloud<pcl::PointXYZ>::Ptr image_points(
      new pcl::PointCloud<pcl::PointXYZ>());
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    cloud_detections::buildKDTree(kdtree, image_points, transformed_cloud,
                                  intrinsic_mat_);

    // Get depth of 2D detections from kdtree
    // depth size = size of bboxes
    depth_vec_ = cloud_detections::computeDepthForBoundingBoxes(kdtree, image_points,
                                                                static_bboxes, k_near_);

    // Get the 3D co-ordinates of 2D detection in base frame
    cam_points = convertPixelsTo3D(static_bboxes, depth_vec_, K_inv_);
  }

  // Handle dynamic bboxes
  std::vector<LShapePose> bboxes_pose;
  if(!dynamic_bboxes.empty())
  {
    if(use_vision_orientation_)
    {
      // Run vision orientation network
      vision_orient_->runInference(init_image_, dynamic_bboxes);

      // Transform pose from cam frame to base frame
      transformLShapeObjects(bboxes_pose);
    }
    else
    {
      // Get the objects 3D pose and orienatation w.r.t
      // camera co-ordiante frame using PCA
      bboxes_pose = cloud_detections::computeBBoxPose(
        transformed_cloud, intrinsic_mat_, bboxes, init_image_.rows, init_image_.cols);

      // Transform pose from cam frame to base frame
      transformLShapeObjects(bboxes_pose);
    }
  }

  // Publish 2D detections and Occupancy grid
  publishObjectDetections(detection_pub_, bboxes, init_image_, resize_);
  publishOccupancyGrid(occ_grid_->grid_map_, base_frame_);

  // Publish object visualizations
  publishObjectVisualizations(bboxes_pose, cam_points, static_bboxes, viz_pub_);
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

    // Convert to 3D point in camera frame
    cam_point = cloud_detections::pixelTo3D(pixel_center, depths[i], K_inv);

    // Transform to base frame
    geometry_msgs::msg::Point base_point
      = transformPointToBaseFrame(cam_point, camera_frame_, base_frame_);

    point_vec.emplace_back(base_point);
  }

  return point_vec;
}

geometry_msgs::msg::Point
GridVision::transformPointToBaseFrame(const geometry_msgs::msg::Point &cam_point,
                                      const std::string &source,
                                      const std::string &target)
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

geometry_msgs::msg::Pose
GridVision::transformPoseToBaseFrame(const geometry_msgs::msg::Pose &obj_pose,
                                     const std::string &source, const std::string &target)
{
  geometry_msgs::msg::Pose base_pose;

  try
  {
    // Lookup transform from camera frame to base frame
    geometry_msgs::msg::TransformStamped transform_stamped
      = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);

    // Transform the point
    tf2::doTransform(obj_pose, base_pose, transform_stamped);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Transform failed: %s", ex.what());
  }

  return base_pose;
}

std::tuple<std::vector<BoundingBox>, std::vector<BoundingBox>>
GridVision::filterBBoxes(const std::vector<BoundingBox> &bboxes)
{
  std::vector<BoundingBox> static_bboxes;
  std::vector<BoundingBox> dynamic_bboxes;

  for(const auto &bbox : bboxes)
  {
    if(bbox.label == ObjectClass::VEHICLE || bbox.label == ObjectClass::BIKE
       || bbox.label == ObjectClass::MOTORBIKE || bbox.label == ObjectClass::PERSON)
    {
      dynamic_bboxes.emplace_back(bbox);
    }
    else
    {
      static_bboxes.emplace_back(bbox);
    }
  }
  return std::make_tuple(static_bboxes, dynamic_bboxes);
}

void GridVision::publishObjectVisualizations(
  const std::vector<LShapePose> &lshape_boxes,
  const std::vector<geometry_msgs::msg::Point> &static_positions,
  const std::vector<BoundingBox> &static_bboxes,
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &marker_pub)
{
  visualization_msgs::msg::MarkerArray marker_array;
  int id = 0;

  // === Visualize static objects (Traffic Lights and Signs) ===
  for(size_t i = 0; i < static_positions.size(); ++i)
  {
    const auto &pos = static_positions[i];
    const auto &static_bbox = static_bboxes[i];

    // Traffic Lights: Draw colored circle
    if(static_bbox.label == ObjectClass::TRAFFIC_LIGHT_RED
       || static_bbox.label == ObjectClass::TRAFFIC_LIGHT_ORANGE
       || static_bbox.label == ObjectClass::TRAFFIC_LIGHT_GREEN)
    {
      visualization_msgs::msg::Marker light_marker;
      light_marker.header.frame_id = base_frame_;
      light_marker.header.stamp = rclcpp::Clock().now();
      light_marker.ns = "traffic_light";
      light_marker.id = id++;
      light_marker.type = visualization_msgs::msg::Marker::SPHERE;
      light_marker.action = visualization_msgs::msg::Marker::ADD;
      light_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
      light_marker.pose.position = pos;
      light_marker.pose.orientation.w = 1.0;
      light_marker.scale.x = 0.3;
      light_marker.scale.y = 0.3;
      light_marker.scale.z = 0.3;

      light_marker.color.a = 1.0;
      if(static_bbox.label == ObjectClass::TRAFFIC_LIGHT_RED)
      {
        light_marker.color.r = 1.0;
      }
      else if(static_bbox.label == ObjectClass::TRAFFIC_LIGHT_ORANGE)
      {
        light_marker.color.r = 1.0;
        light_marker.color.g = 1.0;
      }
      else if(static_bbox.label == ObjectClass::TRAFFIC_LIGHT_GREEN)
      {
        light_marker.color.g = 1.0;
      }

      marker_array.markers.push_back(light_marker);
    }

    // Traffic Signs: Draw numbers
    if(static_bbox.label == ObjectClass::TRAFFIC_SIGN_30
       || static_bbox.label == ObjectClass::TRAFFIC_SIGN_60
       || static_bbox.label == ObjectClass::TRAFFIC_SIGN_90)
    {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = base_frame_;
      text_marker.header.stamp = rclcpp::Clock().now();
      text_marker.ns = "traffic_sign";
      text_marker.id = id++;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
      text_marker.pose.position = pos;
      text_marker.pose.position.z += 1.0;
      text_marker.pose.orientation.w = 1.0;

      text_marker.scale.z = 0.5;
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      text_marker.color.a = 1.0;

      if(static_bbox.label == ObjectClass::TRAFFIC_SIGN_30)
      {
        text_marker.text = "SPEED LIMIT: 30 KMPH";
      }
      else if(static_bbox.label == ObjectClass::TRAFFIC_SIGN_60)
      {
        text_marker.text = "SPEED LIMIT: 60 KMPH";
      }
      else if(static_bbox.label == ObjectClass::TRAFFIC_SIGN_90)
      {
        text_marker.text = "SPEED LIMIT: 90 KMPH";
      }

      marker_array.markers.push_back(text_marker);
    }
  }

  // === Visualize L-shape 3D Bounding Boxes ===
  for(const auto &box : lshape_boxes)
  {
    visualization_msgs::msg::Marker box_marker;
    box_marker.header.frame_id = base_frame_;
    box_marker.header.stamp = rclcpp::Clock().now();
    box_marker.ns = "lshape_bbox";
    box_marker.id = id++;
    box_marker.type = visualization_msgs::msg::Marker::CUBE;
    box_marker.action = visualization_msgs::msg::Marker::ADD;
    box_marker.lifetime = rclcpp::Duration::from_seconds(0.2);

    box_marker.pose = box.pose;
    box_marker.scale.x = box.length;
    box_marker.scale.y = box.width;
    box_marker.scale.z = 2.0;

    box_marker.color.r = 0.0;
    box_marker.color.g = 0.5;
    box_marker.color.b = 1.0;
    box_marker.color.a = 1.0;

    marker_array.markers.push_back(box_marker);
  }

  marker_pub->publish(marker_array);
}

void GridVision::transformLShapeObjects(std::vector<LShapePose> &bboxes_pose)
{
  for(auto &lshape : bboxes_pose)
  {
    lshape.pose = transformPoseToBaseFrame(lshape.pose, camera_frame_, base_frame_);
  }
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GridVision>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
