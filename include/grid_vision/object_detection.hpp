#ifndef OBJECT_DETECTION__OBJECT_DETECTION_HPP_
#define OBJECT_DETECTION__OBJECT_DETECTION_HPP_

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_resource.h>

#include <spdlog/spdlog.h>
#include <sys/types.h>

struct BoundingBox
{
  float x, y, width, height, confidence;
  int class_id;
};

namespace object_detection
{
  // Image preprocessing function
  cv::Mat preprocess_image(const cv::Mat &, uint16_t, uint16_t);

  // Convert OpenCV Mat to ONNX tensor format
  std::vector<float> mat_to_tensor(const cv::Mat &);

  std::unique_ptr<Ort::Session>
  initialize_onnx_runtime(Ort::Env &, Ort::SessionOptions &, const char *);

  std::vector<Ort::Value>
  run_inference(const std::vector<float> &, const std::unique_ptr<Ort::Session> &);

  std::vector<BoundingBox> extract_bboxes(Ort::Value &, double);

  void draw_bboxes(cv::Mat &, const std::vector<BoundingBox> &);

  cv::Mat setIntrinsicMatrix(double, double, double, double);
}
#endif // OBJECT_DETECTION__OBJECT_DETECTION_HPP_
