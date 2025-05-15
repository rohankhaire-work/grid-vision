#ifndef DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
#define DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity <= Severity::kINFO)
      std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

class MonoDepthEstimation
{
public:
  MonoDepthEstimation(int, int, const std::string &);
  void runInference(std::unique_ptr<nvinfer1::IExecutionContext> &context,
                    const cv::Mat &input_img);

private:
  int resize_h_, resize_w_;
  void *buffers[2];
  float *input_host = nullptr;
  float *output_host = nullptr;
  cudaStream_t stream;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  nvinfer1::RunOptions options;

  cv::Mat preprocessImage(const cv::Mat &image, int input_width, int input_height);
  std::vector<float> imageToTensor(const cv::Mat &);
  void initializeTRT(const std::string &engine_file);
};

#endif // DEPTH_ESTIMATION__DEPTH_ESTIMATION_HPP_
