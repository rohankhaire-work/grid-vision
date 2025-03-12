#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/Image.hpp>

namespace object_detection {
// Image preprocessing function
cv::Mat preprocess_image(const cv::Mat, int, int);

// Convert OpenCV Mat to ONNX tensor format
std::vector<float> mat_to_tensor(const cv::Mat);

} // namespace object_detection
