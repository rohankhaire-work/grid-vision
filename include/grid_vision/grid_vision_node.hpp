#ifndef GRID_VISION_NODE__GRID_VISION_NODE_HPP_
#define GRID_VISION_NODE__GRID_VISION_NODE_HPP_

#include "grid_vision/object_detection.hpp"
#include "grid_vision/cloud_detections.hpp"
#include "grid_vision/occupancy_grid.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <optional>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <image_transport/image_transport.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <vector>
#include <string>
#include <chrono>

class GridVision : public rclcpp::Node
{
public:
  GridVision();

private:
  // Params
  std::string image_topic_;
  std::string lidar_topic_;
  std::string det_weight_file_;
  std::string lidar_frame_;
  std::string camera_frame_;
  std::string base_frame_;
  double conf_threshold_;
  double iou_threshold_;
  uint16_t resize_;
  int cam_height_, cam_width_;
  double fx_, fy_, cx_, cy_;
  uint16_t k_near_;
  uint8_t grid_x_, grid_y_;
  double resolution_;

  // Variables
  cv::Mat init_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  Eigen::Matrix3d intrinsic_mat_;
  Eigen::Matrix3d K_inv_;
  sensor_msgs::msg::PointCloud2 init_cloud_;
  pcl::PointCloud<pcl::PointXYZI> cloud_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::optional<OccupancyGridMap> occ_grid_;
  std::vector<float> depth_vec_;

  // ONNX
  std::unique_ptr<Ort::Session> session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;

  // Subscriber
  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  image_transport::Publisher detection_pub_;
  image_transport::Publisher depth_img_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub_;

  void timerCallback();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &);
  void publishObjectDetections(const image_transport::Publisher &,
                               std::vector<BoundingBox> &, cv::Mat &, int);
  void publishOccupancyGrid(const grid_map::GridMap &grid_map, const std::string &base);

  pcl::PointCloud<pcl::PointXYZI>::Ptr
  transformLidarToCamera(const pcl::PointCloud<pcl::PointXYZI> &, const std::string &,
                         const std::string &);

  geometry_msgs::msg::Point
  transformToBaseFrame(const geometry_msgs::msg::Point &, const std::string &,
                       const std::string &);

  std::vector<geometry_msgs::msg::Point>
  convertPixelsTo3D(const std::vector<BoundingBox> &, const std::vector<float> &,
                    const Eigen::Matrix3d &);
};

#endif // GRID_VISION_NODE__GRID_VISION_NODE_HPP_
