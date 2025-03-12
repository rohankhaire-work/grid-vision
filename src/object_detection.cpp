#include "grid_vision/object_detection.hpp"

cv::Mat preprocess_image(const cv::Mat &image, int input_width,
                         int input_height) {
  cv::Mat resized, float_image;

  // Resize to model input size
  cv::resize(image, resized, cv::Size(input_width, input_height));

  // Convert to float32
  resized.convertTo(float_image, CV_32F, 1.0 / 255.0); // Normalize to [0,1]

  // Convert from HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  cv::Mat chw_image;
  cv::vconcat(channels, chw_image); // Stack channels in CHW order

  return chw_image;
}

// Convert OpenCV Mat to ONNX tensor format
std::vector<float> mat_to_tensor(const cv::Mat &mat) {
  std::vector<float> tensor_data;
  if (mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else {
    for (int i = 0; i < mat.rows; i++)
      tensor_data.insert(tensor_data.end(), mat.ptr<float>(i),
                         mat.ptr<float>(i) + mat.cols);
  }
  return tensor_data;
}
