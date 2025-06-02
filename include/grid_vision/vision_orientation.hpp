#ifndef VISION_OREINTATION__VISION_ORIENTATION_HPP_
#define VISION_OREINTATION__VISION_ORIENTATION_HPP_

#include "grid_vision/object_detection.hpp"
#include "grid_vision/cloud_detections.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <span>

struct CAMParams
{
  int network_h;
  int network_w;
  int orig_h;
  int orig_w;
  float fx, fy, cx, cy;
};

struct Vec3
{
  double x, y, z;
};

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
};

class VisionOrientation
{
public:
  VisionOrientation(const CAMParams &, const std::string &);
  ~VisionOrientation();

  std::vector<LShapePose> runInference(const cv::Mat &, const std::vector<BoundingBox> &);

private:
  int resize_h_, resize_w_, orig_w_, orig_h_;
  Logger gLogger;
  std::vector<float> result_;
  const int max_batch_size_ = 8;
  std::vector<float> angle_bins_;
  Eigen::Matrix<float, 4, 4> proj_mat_;

  // Buffers
  void *buffers_[4];
  float *input_host_ = nullptr;
  float *output_orientation_ = nullptr;
  float *output_conf_ = nullptr;
  float *output_dims_ = nullptr;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream_;

  void setProjMat(float, float, float, float);
  cv::Mat normalizeRGB(const cv::Mat &);
  std::vector<float> preprocessImage(const cv::Mat &, const std::vector<BoundingBox> &);
  std::vector<float> imageToTensor(const cv::Mat &);
  void initializeTRT(const std::string &);
  cv::Mat getNetworkBoundingBox(const cv::Mat &, const BoundingBox &);
  std::vector<float> generateBins(int);
  float computeAlpha(const std::span<const float> &, const std::span<const float> &, int);
  float computeThetaRay(const BoundingBox &);
  geometry_msgs::msg::Pose calcLocation(const std::span<const float> &dimension,
                                        const BoundingBox &, float, float);
  std::vector<LShapePose> postProcessOutputs(const std::span<const float> &orient_batch,
                                             const std::span<const float> &conf_batch,
                                             const std::span<const float> &dims_batch,
                                             const std::vector<BoundingBox> &bboxes);
  Eigen::Matrix3f rotationMatrix(float);
};

#endif // VISION_ORIENTATION__VISION_ORIENTATION_HPP_
