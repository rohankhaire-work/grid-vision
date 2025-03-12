#ifndef GRID_VISION_NODE__GRID_VISION_NODE_HPP_
#define GRID_VISION_NODE__GRID_VISION_NODE_HPP_

#include "grid_vision/object_detection.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>

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
  double conf_threshold_;
  double iou_threshold_;
  uint resize_;

  // ONNX
  std::unique_ptr<Ort::Session> session_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;

  // Subscriber
  image_transport::Subscriber image_sub_;

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
};

#endif // GRID_VISION_NODE__GRID_VISION_NODE_HPP_
