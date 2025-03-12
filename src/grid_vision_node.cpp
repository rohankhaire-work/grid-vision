#include "grid_vision/grid_vision_node.hpp"

GridVision::GridVision() : Node("grid_vision_node")
{
  // Set parameters
  image_topic_ = declare_parameter<std::string>("image_topic", "");
  weight_file_ = declare_parameter<std::string>("weights_file", "");
  conf_threshold_ = declare_parameter("confidence_threshold", 0.5);
  iou_threshold_ = declare_parameter("iou_threshold", 0.4);
  resize_ = declare_parameter("network_input_size", 416);

  if(image_topic_.empty() || weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Check if topic name or weight file is assigned");
    return;
  }

  // Image Transport for subscribing
  image_transport::ImageTransport it(shared_from_this());
  image_sub_ = it.subscribe(
    image_topic_, 1, std::bind(&GridVision::imageCallback, this, std::placeholders::_1));

  // Initialize ONNX runtim
  session_ = object_detection::initialize_onnx_runtime(env_, session_options_,
                                                       weight_file_.c_str());
}

void GridVision::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  // Convert ROS2 image message to OpenCV format
  cv::Mat image = cv_bridge::toCvCopy(msg, "rgb8")->image;
  if(image.empty())
    return;

  // Preprocess the image
  // Convert it to tensors
  cv::Mat preprocessed = object_detection::preprocess_image(image, resize_, resize_);
  std::vector<float> input_tensor = object_detection::mat_to_tensor(preprocessed);

  // Perform inference
  std::vector<float> output = object_detection::run_inference(input_tensor, session_);
}

int main(int argc, char *argv[])
{
  auto node = GridVision();
  return 0;
}
