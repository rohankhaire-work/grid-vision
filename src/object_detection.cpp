#include "grid_vision/object_detection.hpp"
#include <cstdint>

namespace object_detection
{
  cv::Mat
  preprocess_image(const cv::Mat &image, uint16_t input_width, uint16_t input_height)
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
  std::vector<Ort::Value> run_inference(const std::vector<float> &image_tensor,
                                        const std::unique_ptr<Ort::Session> &session)
  {
    try
    {
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

      return output_tensors;
    }
    catch(const std::exception &e)
    {
      spdlog::error("Inference error: %s", e.what());
      return {};
    }
  }

  std::vector<BoundingBox>
  extract_bboxes(Ort::Value &output_tensor, double conf_threshold)
  {
    std::vector<BoundingBox> boxes;

    float *raw_output = output_tensor.GetTensorMutableData<float>();
    auto shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    int num_detections = shape[0]; // Number of detected objects
    int num_features = shape[1];   // Number of output values per detection

    for(int i = 0; i < num_detections; i++)
    {
      float x_center = raw_output[i * num_features + 0];
      float y_center = raw_output[i * num_features + 1];
      float width = raw_output[i * num_features + 2];
      float height = raw_output[i * num_features + 3];
      float confidence = raw_output[i * num_features + 4];

      if(confidence < conf_threshold)
        continue; // Ignore low-confidence detections

      // Find the class with the highest probability
      int class_id = -1;
      float max_prob = 0.0;
      for(int j = 5; j < num_features; j++)
      {
        if(raw_output[i * num_features + j] > max_prob)
        {
          max_prob = raw_output[i * num_features + j];
          class_id = j - 5; // Class index
        }
      }

      // Get Enum Class
      ObjectClass label = getObjectClass(class_id);

      // Convert center-based to top-left-based bbox
      float x = x_center - width / 2.0;
      float y = y_center - height / 2.0;

      boxes.push_back({x, y, width, height, confidence, label});
    }

    return boxes;
  }

  void draw_bboxes(cv::Mat &image, const std::vector<BoundingBox> &bboxes)
  {
    for(const auto &box : bboxes)
    {
      cv::Rect rect(box.x, box.y, box.width, box.height);
      cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
      std::string label
        = objectClassToString(box.label) + " (" + std::to_string(box.confidence) + ")";
      cv::putText(image, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0), 1);
    }
  }

  cv::Mat setIntrinsicMatrix(double fx, double fy, double cx, double cy)
  {
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    return K;
  }

  cv::Mat computeKInverse(const cv::Mat &K)
  {
    cv::Mat K_inv;
    cv::invert(K, K_inv, cv::DECOMP_LU); // Compute the inverse using LU decomposition
    return K_inv;
  }

  // Function to convert an integer label to ObjectClass enum
  ObjectClass getObjectClass(int label)
  {
    switch(label)
    {
    case 0: return ObjectClass::BIKE;
    case 1: return ObjectClass::MOTORBIKE;
    case 2: return ObjectClass::PERSON;
    case 3: return ObjectClass::TRAFFIC_LIGHT_GREEN;
    case 4: return ObjectClass::TRAFFIC_LIGHT_ORANGE;
    case 5: return ObjectClass::TRAFFIC_LIGHT_RED;
    case 6: return ObjectClass::TRAFFIC_SIGN_30;
    case 7: return ObjectClass::TRAFFIC_SIGN_60;
    case 8: return ObjectClass::TRAFFIC_SIGN_90;
    case 9: return ObjectClass::VEHICLE;
    }
  }

  // Function to print object class
  std::string objectClassToString(ObjectClass objClass)
  {
    switch(objClass)
    {
    case ObjectClass::BIKE: return "Bike";
    case ObjectClass::MOTORBIKE: return "Motorbike";
    case ObjectClass::PERSON: return "Person";
    case ObjectClass::TRAFFIC_LIGHT_GREEN: return "Light Green";
    case ObjectClass::TRAFFIC_LIGHT_ORANGE: return "Light Orange";
    case ObjectClass::TRAFFIC_LIGHT_RED: return "Light Red";
    case ObjectClass::TRAFFIC_SIGN_30: return "Sign 30";
    case ObjectClass::TRAFFIC_SIGN_60: return "Sign 60";
    case ObjectClass::TRAFFIC_SIGN_90: return "Sign 90";
    case ObjectClass::VEHICLE: return "Vehicle";
    }
  }
};
