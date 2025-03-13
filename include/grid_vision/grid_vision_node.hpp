#ifndef GRID_VISION_NODE__GRID_VISION_NODE_HPP_
#define GRID_VISION_NODE__GRID_VISION_NODE_HPP_

#include "grid_vision/object_detection.hpp"
#include "grid_vision/cloud_detections.hpp"

#include <cstdint>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/hal/interface.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <image_transport/image_transport.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <vector>

class GridVision : public rclcpp::Node
{
public:
  GridVision();

private:
  // Params
  std::string image_topic_;
  std::string weight_file_;
  std::string lidar_frame_;
  std::string camera_frame_;
  double conf_threshold_;
  double iou_threshold_;
  uint16_t resize_;
  double fx_, fy_;
  uint16_t cx_, cy_, k_near_;

  // Variables
  cv::Mat init_image_;
  cv::Mat intrinsic_mat_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

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

  void timerCallback();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &);
  void publishObjectDetections(const image_transport::Publisher &,
                               std::vector<BoundingBox> &, cv::Mat &);
  pcl::PointCloud<pcl::PointXYZ>::Ptr
  transformLidarToCamera(const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const std::string &,
                         const std::string &);
};

#endif // GRID_VISION_NODE__GRID_VISION_NODE_HPP_
