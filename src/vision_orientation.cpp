#include "grid_vision/vision_orientation.hpp"
#include "grid_vision/object_detection.hpp"

#include <cmath>
#include <algorithm>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

VisionOrientation::VisionOrientation(const CAMParams &cam_params,
                                     const std::string &weight_file)
{
  // Set depth img size and detection img size
  resize_h_ = cam_params.network_h;
  resize_w_ = cam_params.network_w;
  orig_h_ = cam_params.orig_h;
  orig_w_ = cam_params.orig_w;

  // Set projection matrix
  proj_mat_ << cam_params.fx, 0.0f, cam_params.cx, 0.0f, 0.0f, cam_params.fy,
    cam_params.cy, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;

  // Set up TRT
  initializeTRT(weight_file);

  // Allocate buffers
  cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                 max_batch_size_ * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&buffers_[0], max_batch_size_ * 3 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_orientation_),
                 max_batch_size_ * 2 * 2 * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_conf_),
                 max_batch_size_ * 2 * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_dims_),
                 max_batch_size_ * 3 * sizeof(float));
  cudaMalloc(&buffers_[1], max_batch_size_ * 2 * 2 * sizeof(float));
  cudaMalloc(&buffers_[2], max_batch_size_ * 2 * sizeof(float));
  cudaMalloc(&buffers_[3], max_batch_size_ * 3 * sizeof(float));

  // Create stream
  cudaStreamCreate(&stream_);

  // Generate bins
  angle_bins_ = generateBins(2);
}

VisionOrientation::~VisionOrientation()
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
  if(buffers_[2])
  {
    cudaFree(buffers_[2]);
    buffers_[2] = nullptr;
  }
  if(buffers_[3])
  {
    cudaFree(buffers_[3]);
    buffers_[3] = nullptr;
  }
  if(input_host_)
  {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if(output_orientation_)
  {
    cudaFreeHost(output_orientation_);
    output_orientation_ = nullptr;
  }
  if(output_conf_)
  {
    cudaFreeHost(output_conf_);
    output_conf_ = nullptr;
  }
  if(output_dims_)
  {
    cudaFreeHost(output_dims_);
    output_dims_ = nullptr;
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
  // Clamp values within image bounds
  int xmin = std::max(0, static_cast<int>(bbox.x_min));
  int ymin = std::max(0, static_cast<int>(bbox.y_min));
  int xmax = std::min(img.cols - 1, static_cast<int>(bbox.x_max));
  int ymax = std::min(img.rows - 1, static_cast<int>(bbox.y_max));

  // Compute width and height
  int width = xmax - xmin;
  int height = ymax - ymin;

  // Crop ROI
  cv::Rect roi(xmin, ymin, width, height);
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

std::vector<LShapePose>
VisionOrientation::runInference(const cv::Mat &input_img,
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
  context->setInputShape("input", inputDims);

  // Set up inference buffers
  context->setInputTensorAddress("input", buffers_[0]);
  context->setOutputTensorAddress("orientation", buffers_[1]);
  context->setOutputTensorAddress("confidence", buffers_[2]);
  context->setOutputTensorAddress("dimension", buffers_[3]);

  // inference
  context->enqueueV3(stream_);

  // Copy the result back
  cudaMemcpyAsync(output_orientation_, buffers_[1], bboxes.size() * 2 * 2 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_conf_, buffers_[2], bboxes.size() * 2 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_dims_, buffers_[3], bboxes.size() * 3 * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  cudaStreamSynchronize(stream_);

  // Use std::span for ease of access
  std::span<float> orientation_span(output_orientation_, bboxes.size() * 2 * 2);
  std::span<float> conf_span(output_conf_, bboxes.size() * 2);
  std::span<float> dims_span(output_dims_, bboxes.size() * 3);

  // Pose process output
  std::vector<LShapePose> bbox_3d
    = postProcessOutputs(orientation_span, conf_span, dims_span, bboxes);

  return bbox_3d;
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
                                      const std::span<const float> &conf, int argmax)
{
  // Get cosine and sine values for that bin
  float cos_val = orient[argmax * 2 + 0];
  float sin_val = orient[argmax * 2 + 1];

  // Compute raw orientation angle
  float alpha = std::atan2(sin_val, cos_val);

  // Adjust with the center of the angle bin
  alpha += angle_bins_[argmax];
  alpha -= static_cast<float>(M_PI);

  return alpha;
}

float VisionOrientation::computeThetaRay(const BoundingBox &bbox)
{
  float fx = proj_mat_(0, 0); // Focal length in x-direction
  float fovx = 2.0f * std::atan(orig_w_ / (2.0f * fx));

  float box_center_x = (bbox.x_min + bbox.x_max) / 2.0f;
  float dx = box_center_x - (orig_w_ / 2.0f);

  float sign = (dx < 0) ? -1.0f : 1.0f;
  dx = std::abs(dx);

  float angle = std::atan((2.0f * dx * std::tan(fovx / 2.0f)) / orig_w_);
  angle *= sign;

  return angle;
}

geometry_msgs::msg::Pose
VisionOrientation::calcLocation(const std::array<double, 3> &dimension,
                                const BoundingBox &bbox, float alpha, float theta_ray)
{
  float orient = alpha + theta_ray;
  Eigen::Matrix3f R = rotationMatrix(orient);

  std::array<float, 4> box_corners
    = {static_cast<float>(bbox.x_min), static_cast<float>(bbox.y_min),
       static_cast<float>(bbox.x_max), static_cast<float>(bbox.y_max)};

  // Dimension halves
  float dx = dimension[0] / 2.0f; // length / 2
  float dy = dimension[1] / 2.0f; // height / 2
  float dz = dimension[2] / 2.0f; // width / 2

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
  std::vector<Vec3> comb(4);
  constraints.reserve(64);
  for(const auto &left : left_constraints)
    for(const auto &top : top_constraints)
      for(const auto &right : right_constraints)
        for(const auto &bottom : bottom_constraints)
        {
          comb[0] = left;
          comb[1] = top;
          comb[2] = right;
          comb[3] = bottom;

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
    std::array<Eigen::Matrix<float, 3, 4>, 4> projected_M_array;

    for(int i = 0; i < 4; ++i)
    {
      M_array[i] = pre_M;

      Eigen::Vector3f RX
        = R * Eigen::Vector3f(constraint[i].x, constraint[i].y, constraint[i].z);
      M_array[i].block<3, 1>(0, 3) = RX;

      projected_M_array[i] = proj_mat_ * M_array[i];
    }
    // Construct A (4x3) and b (4x1)
    Eigen::Matrix<float, 4, 3> A;
    Eigen::Matrix<float, 4, 1> b;

    for(int row = 0; row < 4; ++row)
    {
      int idx = indices[row];
      const Eigen::Matrix<float, 3, 4> &M = projected_M_array[row];
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

  geometry_msgs::msg::Pose best_pose;

  best_pose.position.x = best_loc[0];
  best_pose.position.y = best_loc[1];
  best_pose.position.z = best_loc[2];

  // Pose (orientation as quaternion)
  tf2::Quaternion q;
  q.setRPY(0, -orient, 0);
  best_pose.orientation.x = q.x();
  best_pose.orientation.y = q.y();
  best_pose.orientation.z = q.z();
  best_pose.orientation.w = q.w();

  return best_pose;
}

std::vector<LShapePose>
VisionOrientation::postProcessOutputs(const std::span<const float> &orient_batch,
                                      const std::span<const float> &conf_batch,
                                      const std::span<const float> &dims_batch,
                                      const std::vector<BoundingBox> &bboxes)
{
  std::vector<LShapePose> bbox_3d;

  for(int i = 0; i < bboxes.size(); ++i)
  {
    LShapePose result;
    // Slice conf and orient for each element
    std::span<const float> conf_sample = conf_batch.subspan(i * 2, 2);
    std::span<const float> orient_sample = orient_batch.subspan(i * 4, 4);
    std::span<const float> dims_sample = dims_batch.subspan(i * 3, 3);

    // Find the index of the maximum confidence
    auto max_it = std::max_element(conf_sample.begin(), conf_sample.end());
    int argmax = static_cast<int>(max_it - conf_sample.begin());

    float alpha = computeAlpha(orient_sample, conf_sample, argmax);
    float theta_ray = computeThetaRay(bboxes[i]);

    if(bboxes[i].label == ObjectClass::VEHICLE)
    {
      result.length = dims_sample[2] + car_avg_len_;
      result.width = dims_sample[0] + car_avg_wid_;
      result.height = dims_sample[1] + car_avg_ht_;
    }
    else if(bboxes[i].label == ObjectClass::BIKE)
    {
      result.length = dims_sample[2] + bicycle_avg_len_;
      result.width = dims_sample[0] + bicycle_avg_wid_;
      result.height = dims_sample[1] + bicycle_avg_ht_;
    }
    else if(bboxes[i].label == ObjectClass::MOTORBIKE)
    {
      result.length = dims_sample[2] + bike_avg_len_;
      result.width = dims_sample[0] + bike_avg_wid_;
      result.height = dims_sample[1] + bike_avg_ht_;
    }
    else if(bboxes[i].label == ObjectClass::PERSON)
    {
      result.length = dims_sample[2] + person_avg_len_;
      result.width = dims_sample[0] + person_avg_wid_;
      result.height = dims_sample[1] + person_avg_ht_;
    }
    else
    {
      continue;
    }

    std::array<double, 3> lwh = {result.length, result.width, result.height};

    geometry_msgs::msg::Pose bbox_pose = calcLocation(lwh, bboxes[i], alpha, theta_ray);

    result.pose = bbox_pose;

    bbox_3d.emplace_back(result);
  }
  return bbox_3d;
}

Eigen::Matrix3f VisionOrientation::rotationMatrix(float theta)
{
  float c = std::cos(theta);
  float s = std::sin(theta);
  Eigen::Matrix3f R;
  R << c, 0, s, 0, 1, 0, -s, 0, c;
  return R;
}
