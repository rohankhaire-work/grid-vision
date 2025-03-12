#ifndef GRID_VISION_NODE__GRID_VISION_NODE_HPP_
#define GRID_VISION_NODE__GRID_VISION_NODE_HPP_

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>

#include <memory>

class GridVision : public rclcpp::Node
{
public:
  GridVision();

private:
  std::string image_topic_;
  std::string weight_file_;
  float conf_threshold_;
  float iou_threshold_;

  image_transport::Subscriber image_sub_;

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
};

#endif // GRID_VISION_NODE__GRID_VISION_NODE_HPP_
