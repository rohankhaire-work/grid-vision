#include "grid_vision/grid_vision_node.hpp"

GridVision() : Node("grid_vision_node")
{
  // Set parameters
  declare_parameter<std::string>(image_topic_, "");
  declare_parameter<std::string>(weight_file_, "");
  declare_parameter<float>(conf_threshold_, 0.5);
  declare_parameter<float>(iou_threshold_, 0.4);

  if(image_topic_.empty() || weight_file_.empty())
    {
      RCLCPP_ERROR(this->get_logger(),
                   "Check if topic name or weight file is assigned");
      return;
    }
  // Image Transport for subscribing
  image_transport::ImageTransport it(shared_from_this());
  image_sub_ = it.subscribe(
    image_topic_, 1,
    std::bind(&GridVision::imageCallback, this, std::placeholders::_1));
}

void GridVision::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  try
    {
      // Convert ROS2 image message to OpenCV format
      cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
    }
  catch(cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
    }
}

int main(int argc, char *argv[])
{
  auto node = GridVision();
  return 0;
}
