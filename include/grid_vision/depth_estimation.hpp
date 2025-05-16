#ifndef DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
#define DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_

#include "grid_vision/object_detection.hpp"

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <geometry_msgs/msg/point.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class MonoDepthEstimation
{
public:
  MonoDepthEstimation(int, int, int, int, int, const std::string &);
  ~MonoDepthEstimation();

  // Delete copy constructor and assignment
  MonoDepthEstimation(const MonoDepthEstimation &) = delete;
  MonoDepthEstimation &operator=(const MonoDepthEstimation &) = delete;

  // Allow move semantics
  MonoDepthEstimation(MonoDepthEstimation &&) noexcept = default;
  MonoDepthEstimation &operator=(MonoDepthEstimation &&) noexcept = default;

  std::vector<float>
  runInference(const cv::Mat &input_img, const std::vector<BoundingBox> &bboxes);
  geometry_msgs::msg::Point
  pixelTo3D(const cv::Point2f &pixel, float depth, const Eigen::Matrix3d &K_inv);

  cv::Mat depth_img_;

private:
  int resize_h_, resize_w_, orig_h_, orig_w_;
  float scale_x_, scale_y_;
  void *buffers_[2];
  float *input_host_ = nullptr;
  float *output_host_ = nullptr;
  cv::Mat depth_map_;
  int patch_size_;
  cudaStream_t stream;
  Logger gLogger;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  cv::Mat preprocessImage(const cv::Mat &, int, int);
  std::vector<float> imageToTensor(const cv::Mat &);
  void initializeTRT(const std::string &);
  std::vector<float> getBoundingBoxDepth(std::vector<BoundingBox>);

  float getCenterPatchDepth(const cv::Mat &, const BoundingBox &, int);

  cv::Mat convertToDepthMap();
  cv::Mat convertToDepthImg();
};

#endif // DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
