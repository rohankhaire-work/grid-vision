#include "grid_vision/vision_orientation.hpp"
#include "grid_vision/cloud_detections.hpp"
#include "grid_vision/object_detection.hpp"

#include <cmath>
#include <algorithm>

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

  // Generate bins
  angle_bins_ = generateBins(2);
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

  // Use std::span for ease of access
  std::span<float> orientation_span(output_orientation_, batch_size * 2 * 2);
  std::span<float> conf_span(output_conf_, batch_size * 2);
  std::span<float> dims_span(output_dims_, batch_size * 3);

  cudaStreamSynchronize(stream_);
}

std::vector<float> VisionOrientation::generateBins(int bins)
{
  std::vector<float> angle_bins(bins, 0.0f);
  float interval = 2.0f * M_PI / bins;

  for(int i = 1; i < bins; ++i)
  {
    angle_bins[i] = i * interval;
  }

  // Add half the interval to shift to bin centers
  for(int i = 0; i < bins; ++i)
  {
    angle_bins[i] += interval / 2.0f;
  }

  return angle_bins;
}

float VisionOrientation::computeAlpha(const std::span<const float> &orient,
                                      const std::span<const float> &conf)
{
  // Find the index of the maximum confidence
  auto it = std::max_element(conf.begin(), conf.end());
  int argmax = static_cast<int>(max_it - conf.begin());

  // Get cosine and sine values for that bin
  float cos_val = orient[argmax * 2 + 0];
  float sin_val = orient[argmax * 2 + 1];

  // Compute raw orientation angle
  float alpha = std::atan2(sin_val, cos_val);

  // Adjust with the center of the angle bin
  alpha += angle_bins[argmax];
  alpha -= static_cast<float>(M_PI);

  return alpha;
}

Result
calc_location(const std::array<float, 3> &dimension, const Eigen::Matrix4f &proj_matrix,
              const Box2D &box_2d, float alpha, float theta_ray)
{
  float orient = alpha + theta_ray;
  Eigen::Matrix3f R = rotation_matrix(orient);

  // Extract box corners
  float xmin = box_2d;
  float ymin = box_2d;
  float xmax = box_2d;
  float ymax = box_2d;

  std::array<float, 4> box_corners = {xmin, ymin, xmax, ymax};

  // Dimension halves
  float dx = dimension[2] / 2.0f; // length / 2
  float dy = dimension[0] / 2.0f; // height / 2
  float dz = dimension[1] / 2.0f; // width / 2

  // Determine multipliers based on alpha
  int left_mult = 1, right_mult = -1;
  const float deg88 = 88 * M_PI / 180.0f;
  const float deg90 = 90 * M_PI / 180.0f;
  const float deg92 = 92 * M_PI / 180.0f;

  if(alpha < deg92 && alpha > deg88)
  {
    left_mult = 1;
    right_mult = 1;
  }
  else if(alpha < -deg88 && alpha > -deg92)
  {
    left_mult = -1;
    right_mult = -1;
  }
  else if(alpha < deg90 && alpha > -deg90)
  {
    left_mult = -1;
    right_mult = 1;
  }

  int switch_mult = (alpha > 0) ? 1 : -1;

  // Build constraints
  std::vector<Vec3> left_constraints;
  std::vector<Vec3> right_constraints;
  std::vector<Vec3> top_constraints;
  std::vector<Vec3> bottom_constraints;

  for(int i : {-1, 1})
  {
    left_constraints.push_back(
      {left_mult * dx, static_cast<float>(i) * dy, -switch_mult * dz});
    right_constraints.push_back(
      {right_mult * dx, static_cast<float>(i) * dy, switch_mult * dz});
  }

  for(int i : {-1, 1})
  {
    for(int j : {-1, 1})
    {
      top_constraints.push_back(
        {static_cast<float>(i) * dx, -dy, static_cast<float>(j) * dz});
      bottom_constraints.push_back(
        {static_cast<float>(i) * dx, dy, static_cast<float>(j) * dz});
    }
  }

  // Generate all 64 combinations of constraints
  std::vector<std::vector<Vec3>> constraints;
  for(const auto &left : left_constraints)
    for(const auto &top : top_constraints)
      for(const auto &right : right_constraints)
        for(const auto &bottom : bottom_constraints)
        {
          std::vector<Vec3> comb = {left, top, right, bottom};
          if(all_unique(comb))
            constraints.push_back(comb);
        }

  // Pre M matrix with 1's on diagonal (4x4)
  Eigen::Matrix4f pre_M = Eigen::Matrix4f::Identity();

  std::array<int, 4> indices = {0, 1, 0, 1}; // correspond to row selector for x or y

  std::array<float, 3> best_loc = {0.f, 0.f, 0.f};
  float best_error = std::numeric_limits<float>::max();
  std::vector<Vec3> best_X;

  for(const auto &constraint : constraints)
  {
    // Create M matrices for each corner
    std::array<Eigen::Matrix4f, 4> M_array;
    for(int i = 0; i < 4; ++i)
    {
      M_array[i] = pre_M;
      Eigen::Vector3f RX
        = R * Eigen::Vector3f(constraint[i][0], constraint[i][1], constraint[i][2]);
      M_array[i].block<3, 1>(0, 3) = RX;
      M_array[i] = proj_matrix * M_array[i];
    }

    // Construct A (4x3) and b (4x1)
    Eigen::Matrix<float, 4, 3> A;
    Eigen::Matrix<float, 4, 1> b;

    for(int row = 0; row < 4; ++row)
    {
      int idx = indices[row];
      const Eigen::Matrix4f &M = M_array[row];
      float box_val = box_corners[row];

      // A row: M[idx, :3] - box_corners[row] * M[2, :3]
      A.row(row) = M.row(idx).head<3>() - box_val * M.row(2).head<3>();

      // b row: box_corners[row]*M[2,3] - M[idx, 3]
      b(row, 0) = box_val * M(2, 3) - M(idx, 3);
    }

    // Solve least squares: loc = (A^T A)^-1 A^T b
    Eigen::Vector3f loc = A.colPivHouseholderQr().solve(b);

    // Calculate error (residual norm)
    float error = (A * loc - b).squaredNorm();

    if(error < best_error)
    {
      best_error = error;
      best_loc = {loc(0), loc(1), loc(2)};
      best_X = constraint;
    }
  }

  return {best_loc, best_X};
}

std::vector<LShapePose>
VisionOrientation::postProcessOutputs(const std::span<const float> &orient_batch,
                                      const std::span<const float> &conf_batch,
                                      const std::span<const float> &dims_batch,
                                      const std::vector<BoundingBox> &bboxes)
{
  std::vector<LShapePose> bbox_pose;

  for(int i = 0; i < batch_size; ++i)
  {
    LShapePose result;
    // Slice conf and orient for each element
    std::span<const float> conf_sample = conf_batch.subspan(i * 2, 2);
    std::span<const float> orient_sample = orient_batch.subspan(i * 4, 4);
    std::span<const float> dims_sample = dims_batch.subspan(i * 3, 3);

    float alpha = computeAlpha(orient_sample, conf_sample);
    float theta_ray = computeThetaRay();
  }
}
