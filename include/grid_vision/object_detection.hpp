#ifndef OBJECT_DETECTION__OBJECT_DETECTION_HPP_
#define OBJECT_DETECTION__OBJECT_DETECTION_HPP_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cuda/cuda_resource.h>

#include <spdlog/spdlog.h>

// Enum for object class labels
enum class ObjectClass
{
  BIKE = 0,
  MOTORBIKE = 1,
  PERSON = 2,
  TRAFFIC_LIGHT_GREEN = 3,
  TRAFFIC_LIGHT_ORANGE = 4,
  TRAFFIC_LIGHT_RED = 5,
  TRAFFIC_SIGN_30 = 6,
  TRAFFIC_SIGN_60 = 7,
  TRAFFIC_SIGN_90 = 8,
  VEHICLE = 9,
  UNKNOWN = 10
};

struct BoundingBox
{
  double x_min, y_min, x_max, y_max;
  float confidence;
  ObjectClass label;
};

namespace object_detection
{
  // Image preprocessing function
  cv::Mat preprocess_image(const cv::Mat &, uint16_t, uint16_t);

  // Convert OpenCV Mat to ONNX tensor format
  std::vector<float> mat_to_tensor(const cv::Mat &);

  void initialize_onnx_runtime(std::unique_ptr<Ort::Session> &, Ort::Env &,
                               Ort::SessionOptions &, const char *);

  std::vector<Ort::Value>
  run_inference(const std::vector<float> &, const std::unique_ptr<Ort::Session> &);

  std::vector<BoundingBox>
  extract_bboxes(const std::vector<Ort::Value> &, double, double, int);

  Eigen::VectorXf computeIoU_Eigen(const BoundingBox &, const Eigen::MatrixXf &);

  std::vector<BoundingBox> fast_non_max_suppression(std::vector<BoundingBox> &, float);

  void denormalizeBoundingBox(std::vector<BoundingBox> &, int);

  void draw_bboxes(cv::Mat &, const std::vector<BoundingBox> &);

  cv::Mat setIntrinsicMatrix(double, double, double, double);

  cv::Mat computeKInverse(const cv::Mat &);

  ObjectClass getObjectClass(int);

  std::string objectClassToString(ObjectClass);

}
#endif // OBJECT_DETECTION__OBJECT_DETECTION_HPP_
