#ifndef OBJECT_DETECTION__OBJECT_DETECTION_HPP_
#define OBJECT_DETECTION__OBJECT_DETECTION_HPP_

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_resource.h>

#include <spdlog/spdlog.h>
#include <sys/types.h>

namespace object_detection
{
  // Image preprocessing function
  cv::Mat preprocess_image(const cv::Mat &, int, int);

  // Convert OpenCV Mat to ONNX tensor format
  std::vector<float> mat_to_tensor(const cv::Mat &);

  std::unique_ptr<Ort::Session>
  initialize_onnx_runtime(Ort::Env &, Ort::SessionOptions &, const char *);

  std::vector<float>
  run_inference(const std::vector<float> &, const std::unique_ptr<Ort::Session> &);

}
#endif // OBJECT_DETECTION__OBJECT_DETECTION_HPP_
