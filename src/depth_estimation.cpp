#include "grid_vision/depth_estimation.hpp"

MonoDepthEstimation::MonoDepthEstimation(int input_h, int input_w,
                                         const std::string &depth_weight_file)
{
  resize_h_ = input_h;
  resize_w_ = input_w;

  // Set up TRT
  initializeTRT(depth_weight_file);

  // Allocate memory in GPU
  cudaMallocHost(input_host, 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers[0], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(output_host, resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers[1], resize_h_ * resize_h_ * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream);
  options.stream = stream;
}

cv::Mat MonoDepthEstimation::preprocessImage(const cv::Mat &image, int input_width,
                                             int input_height)
{
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

std::vector<float> MonoDepthEstimation::imageToTensor(const cv::Mat &mat)
{
  std::vector<float> tensor_data;
  if(mat.isContinuous())
    tensor_data.assign((float *)mat.datastart, (float *)mat.dataend);
  else
  {
    for(int i = 0; i < mat.rows; i++)
      tensor_data.insert(tensor_data.end(), mat.ptr<float>(i),
                         mat.ptr<float>(i) + mat.cols);
  }
  return tensor_data;
}

void MonoDepthEstimation::initializeTRT(const std::string &engine_file)
{
  // Load TensorRT engine from file
  std::ifstream file(engine_file, std::ios::binary);
  if(!file)
  {
    throw std::runtime_error("Failed to open engine file: " + engine_file);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  // Create runtime and deserialize engine
  // Create TensorRT Runtime
  runtime.reset(nvinfer1::createInferRuntime(gLogger));

  // Deserialize engine
  engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context.reset(engine->createExecutionContext());
}

void MonoDepthEstimation::runInference(
  std::unique_ptr<nvinfer1::IExecutionContext> &context, const cv::Mat &input_img)
{
  // Preprocess image and convert to vector
  cv::Mat processed_img = preprocessImage(input_img, resize_w_, resize_h_);
  std::vector<int> input_tensor = imageToTensor(processed_img);

  // Copy to host memory and then to GPU
  std::memcpy(input_host, input_tensor.data(),
              1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(&buffers[0], input_host, 1 * 3 * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  // Set up inference buffers
  context->setInputTensorAddress("input", &buffers[0]);
  context->setOutputTensorAddress("depth", &buffers[1]);

  // inference
  context->enqueueV3(options);

  // Copy the result back
  cudaMemcpyAsync(output_host, &buffers[1], resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
}
