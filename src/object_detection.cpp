#include "grid_vision/object_detection.hpp"
#include <memory>
#include <spdlog/spdlog.h>

namespace object_detection
{
  cv::Mat preprocess_image(const cv::Mat &image, int input_width, int input_height)
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

  // Convert OpenCV Mat to ONNX tensor format
  std::vector<float> mat_to_tensor(const cv::Mat &mat)
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

  std::unique_ptr<Ort::Session>
  initialize_onnx_runtime(Ort::Env &env, Ort::SessionOptions &options,
                          const char *weight_file)
  {
    try
    {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = 0;
      options.SetIntraOpNumThreads(1);
      options.AppendExecutionProvider_CUDA(cuda_options);

      auto session = std::make_unique<Ort::Session>(env, weight_file, options);
      spdlog::info("ONNX Model successfully loaded on GPU!");
      return session;
    }
    catch(const std::exception &e)
    {
      spdlog::error("Failed to initialize ONNX Runtime: %s", e.what());
      return nullptr;
    }
  }

  // **3️⃣ Run Inference on Preprocessed Image**
  std::vector<float> run_inference(const std::vector<float> &image_tensor,
                                   const std::unique_ptr<Ort::Session> &session)
  {
    try
    {
      // Convert OpenCV Mat → ONNX Tensor
      // Batch, Channels, Height, Width
      std::vector<int64_t> input_shape
        = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

      // Create ONNX Tensor
      Ort::AllocatorWithDefaultOptions allocator;
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator, input_shape.data(), input_shape.size());

      // Copy Data
      float *tensor_data = input_tensor.GetTensorMutableData<float>();
      std::memcpy(tensor_data, image_tensor.data(), image_tensor.size() * sizeof(float));

      // Run ONNX Inference
      const char *input_names[] = {"input"};
      const char *output_names[] = {"output"};
      auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names,
                                         &input_tensor, 1, output_names, 1);

      // Extract Results
      float *output_data = output_tensors.front().GetTensorMutableData<float>();
      return std::vector<float>(output_data,
                                output_data + 10); // Assuming 10 output classes
    }
    catch(const std::exception &e)
    {
      spdlog::error("Inference error: %s", e.what());
      return {};
    }
  }
};
