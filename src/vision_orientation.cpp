#include "grid_vision/vision_orientation.hpp"
#include "grid_vision/object_detection.hpp"

VisionOrientation::VisionOrientation(const CAMParams &cam_params,
                                     const std::string &weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = cam_params.network_h;
  resize_w_ = cam_params.network_w;

  // Set up TRT
  initializeTRT(weight_file);

  // Allocate buffers
  cudaError_t err
    = cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                     max_batch_size_ * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], max_batch_size_ * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[1], max_batch_size_ * 2 * 2 * sizeof(float));
  cudaMalloc(&buffers_[2], max_batch_size_ * 2 * sizeof(float));
  cudaMalloc(&buffers_[3], max_batch_size_ * 3 * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream_);
}

MonoDepthEstimation::~MonoDepthEstimation()
{
  if(buffers_[0])
  {
    cudaFree(buffers_[0]);
    buffers_[0] = nullptr;
  }
  if(buffers_[1])
  {
    cudaFree(buffers_[1]);
    buffers_[1] = nullptr;
  }
  if(input_host_)
  {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if(output_host_)
  {
    cudaFreeHost(output_host_);
    output_host_ = nullptr;
  }
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}

cv::Mat VisionOrientation::normalizeRGB(const cv::Mat &input)
{
  std::vector<cv::Mat> channels(3);
  cv::split(input, channels);

  std::vector<cv::Mat> temp_data;
  temp_data.resize(3);

  for(int i = 0; i < 3; ++i)
  {
    cv::Mat float_channel;
    channels[i].convertTo(float_channel, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(float_channel, mean, stddev);

    // Normalize: (x - mean) / std
    temp_data[i] = (float_channel - mean[0]) / stddev[0];
  }

  // Convert to cv::Mat
  cv::Mat normalized_rgb;
  cv::vconcat(temp_data, normalized_rgb);

  return normalized_rgb;
}

std::vector<float>
VisionOrientation::preprocessImage(const cv::Mat &image,
                                   const std::vector<BoundingBox> &bboxes)
{
  std::vector<float> all_bbox_data;
  for(const auto &bbox : bboxes)
  {
    // Get Network ready bbox image
    cv::Mat netw_bbox = getNetworkBoundingBox(image, bbox);

    // Flatten and append
    std::vector<float> img_data(netw_bbox.total() * netw_bbox.channels());
    std::memcpy(img_data.data(), netw_bbox.data,
                netw_bbox.total() * netw_bbox.elemSize());
    all_bbox_data.insert(all_bbox_data.end(), img_data.begin(), img_data.end());
  }

  return all_bbox_data;
}

cv::Mat
VisionOrientation::getNetworkBoundingBox(const cv::Mat &img, const BoundingBox &bbox)
{
  // Compute width and height
  int width = bbox.x_max - bbox.x_min;
  int height = bbox.y_max - bbox.y_min;

  // Crop ROI
  cv::Rect roi(bbox.x_min, bbox.y_min, width, height);
  cv::Mat cropped = img(roi);

  // Resize to Network input dim
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(resize_w_, resize_h_));

  // Normalize
  cv::Mat norm_bbox = normalizeRGB(resized);

  return norm_bbox;
}

void VisionOrientation::initializeTRT(const std::string &engine_file)
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

void VisionOrientation::runInference(const cv::Mat &input_img,
                                     const std::vector<BoundingBox> &bboxes)
{
  // Preprocess image and convert to vector
  std::vector<float> input_tensor = preprocessImage(input_img, bboxes);

  // Copy to host memory and then to GPU
  std::memcpy(input_host_, input_tensor.data(),
              bboxes.size() * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(buffers_[0], input_host_,
                  bboxes.size() * 3 * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Set input dimension
  nvinfer1::Dims4 inputDims(bboxes.size(), 3, resize_h_, resize_w_);
  int input_index = engine->getBindingIndex("input");
  context->setBindingDimensions(input_index, inputDims);

  // Set up inference buffers
  context->setInputTensorAddress("input", buffers_[0]);
  context->setOutputTensorAddress("orientation", buffers_[1]);
  context->setOutputTensorAddress("confidence", buffers_[2]);
  context->setOutputTensorAddress("dims", buffers_[3]);

  // inference
  context->enqueueV3(stream_);

  // Copy the result back
  cudaMemcpyAsync(output_orientation_, buffers_[1], bboxes.size() * 2 * 2 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_conf_, buffers_[2], bboxes.size() * 2 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_dims_, buffers_[1], bboxes.size() * 3 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  cudaStreamSynchronize(stream_);
}
