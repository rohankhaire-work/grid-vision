#include "grid_vision/depth_estimation.hpp"

MonoDepthEstimation::MonoDepthEstimation(int input_h, int input_w, int orig_h, int orig_w,
                                         int patch_size,
                                         const std::string &depth_weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = input_h;
  resize_w_ = input_w;
  orig_h_ = orig_h;
  orig_w_ = orig_w;
  patch_size_ = patch_size;

  // setting up scale
  scale_x_ = static_cast<float>(resize_w_) / orig_w_;
  scale_y_ = static_cast<float>(resize_h_) / orig_h_;

  // Set up TRT
  initializeTRT(depth_weight_file);

  // Allocate memory in GPU
  cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], 1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_host_),
                 resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[1], resize_h_ * resize_h_ * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream);
}

MonoDepthEstimation::~MonoDepthEstimation()
{
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

std::vector<float>
MonoDepthEstimation::runInference(const cv::Mat &input_img,
                                  const std::vector<BoundingBox> &bboxes)
{
  // Preprocess image and convert to vector
  cv::Mat processed_img = preprocessImage(input_img, resize_w_, resize_h_);
  std::vector<float> input_tensor = imageToTensor(processed_img);

  // Copy to host memory and then to GPU
  std::memcpy(input_host_, input_tensor.data(),
              1 * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMemcpyAsync(&buffers_[0], input_host_,
                  1 * 3 * resize_h_ * resize_w_ * sizeof(float), cudaMemcpyHostToDevice,
                  stream);

  // Set up inference buffers
  context->setInputTensorAddress("input", &buffers_[0]);
  context->setOutputTensorAddress("depth", &buffers_[1]);

  // inference
  context->enqueueV3(stream);

  // Copy the result back
  cudaMemcpyAsync(output_host_, &buffers_[1], resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // store the depth image
  depth_img_ = convertToDepthImg();
  depth_map_ = convertToDepthMap();

  // Get the depth of bboxes
  std::vector<float> bboxes_depth = getBoundingBoxDepth(bboxes);

  return bboxes_depth;
}

std::vector<float>
MonoDepthEstimation::getBoundingBoxDepth(std::vector<BoundingBox> bboxes)
{
  std::vector<float> bboxes_depth;
  for(auto &bbox : bboxes)
  {
    // Resize the bbox
    bbox.x_min = bbox.x_min * scale_x_;
    bbox.x_max = bbox.x_max * scale_x_;
    bbox.y_min = bbox.y_min * scale_y_;
    bbox.y_max = bbox.y_max * scale_y_;

    // Get the depth of bbox
    float bbox_depth = getCenterPatchDepth(depth_map_, bbox, patch_size_);

    // Store it
    bboxes_depth.emplace_back(bbox_depth);
  }

  return bboxes_depth;
}

cv::Mat MonoDepthEstimation::convertToDepthImg()
{
  cv::Mat depthMap(resize_h_, resize_w_, CV_32FC1, output_host_);
  cv::Mat depthVis;
  cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX);
  depthVis.convertTo(depthVis, CV_8UC1);

  return depthVis;
}

cv::Mat MonoDepthEstimation::convertToDepthMap()
{
  cv::Mat depth_map(resize_h_, resize_w_, CV_32FC1, output_host_);
  return depth_map;
}

float MonoDepthEstimation::getCenterPatchDepth(const cv::Mat &depth_map,
                                               const BoundingBox &box, int patch_size = 5)
{
  int cx = static_cast<int>((box.x_min + box.x_max) / 2);
  int cy = static_cast<int>((box.y_min + box.y_max) / 2);

  int half_patch = patch_size / 2;

  std::vector<float> depths;

  for(int dy = -half_patch; dy <= half_patch; ++dy)
  {
    for(int dx = -half_patch; dx <= half_patch; ++dx)
    {
      int x = cx + dx;
      int y = cy + dy;

      if(x >= 0 && x < depth_map.cols && y >= 0 && y < depth_map.rows)
      {
        float d = depth_map.at<float>(y, x);
        if(d > 0 && std::isfinite(d))
        {
          depths.push_back(d);
        }
      }
    }
  }

  // No valid depth found
  if(depths.empty())
    return -1.0f;

  // Take the median of depths
  std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
  return depths[depths.size() / 2];
}

geometry_msgs::msg::Point
MonoDepthEstimation::pixelTo3D(const cv::Point2f &pixel, float depth,
                               const Eigen::Matrix3d &K_inv)
{
  // Convert pixel coordinates to homogeneous coordinates
  Eigen::Vector3d pixel_homogeneous(pixel.x, pixel.y, 1.0);
  // Compute the 3D point in camera frame: X_cam = K_inv * (u, v, 1) * depth
  Eigen::Vector3d point_3D = depth * (K_inv * pixel_homogeneous);

  geometry_msgs::msg::Point cam_point;
  cam_point.x = point_3D.x();
  cam_point.y = point_3D.y();
  cam_point.z = point_3D.z();

  return cam_point;
}
