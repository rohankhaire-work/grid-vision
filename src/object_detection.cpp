#include "grid_vision/object_detection.hpp"
#include <Eigen/src/Core/Matrix.h>

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

  void initialize_onnx_runtime(std::unique_ptr<Ort::Session> &session, Ort::Env &env,
                               Ort::SessionOptions &options, const char *weight_file)
  {
    try
    {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = 0;
      options.SetIntraOpNumThreads(1);
      options.AppendExecutionProvider_CUDA(cuda_options);

      session = std::make_unique<Ort::Session>(env, weight_file, options);
      spdlog::info("ONNX Model successfully loaded on GPU!");
    }
    catch(const std::exception &e)
    {
      spdlog::error("Failed to initialize ONNX Runtime: {}", e.what());
    }
  }

  // Run Inference on Preprocessed Image
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
      const char *output_names[] = {"boxes", "confs"};
      auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names,
                                         &input_tensor, 1, output_names, 2);

      return output_tensors;
    }
    catch(const std::exception &e)
    {
      spdlog::error("Inference error: {}", e.what());
      return {};
    }
  }

  // Optimized extraction using Eigen for vectorization
  std::vector<BoundingBox>
  extract_bboxes(const std::vector<Ort::Value> &output_tensors, double conf_threshold,
                 double iou_threshold, int orig_w, int orig_h, int resize)
  {
    std::vector<BoundingBox> bboxes;
    std::vector<BoundingBox> nms_bboxes;

    // Get tensor shapes
    auto shape_boxes = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto shape_scores = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int num_detections = shape_boxes[1]; // 2535
    int num_classes = shape_scores[2];   // 10

    // Access raw data
    const float *boxes_data = output_tensors[0].GetTensorData<float>();
    const float *scores_data = output_tensors[1].GetTensorData<float>();

    // Convert to Eigen maps for vectorized operations
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>>
      boxes_matrix(boxes_data, num_detections, 4);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      scores_matrix(scores_data, num_detections, num_classes);

    for(int i = 0; i < num_detections; i++)
    {
      // Find the class with max confidence using Eigen
      Eigen::Index best_class;
      float max_conf = scores_matrix.row(i).maxCoeff(&best_class);

      // Apply confidence threshold
      if(max_conf >= conf_threshold)
      {
        BoundingBox bbox;
        bbox.confidence = max_conf;
        bbox.label = getObjectClass(static_cast<int>(best_class));

        // Faster bounding box conversion (Vectorized)
        bbox.x_min = boxes_matrix(i, 0);
        bbox.y_min = boxes_matrix(i, 1);
        bbox.x_max = boxes_matrix(i, 2);
        bbox.y_max = boxes_matrix(i, 3);

        bboxes.push_back(bbox);
      }
    }

    // Apply Fast NMS and denormalize the bboxes
    nms_bboxes = fast_non_max_suppression(bboxes, iou_threshold);
    denormalizeAndScaleBoundingBox(nms_bboxes, orig_w, orig_h, resize);

    return nms_bboxes;
  }

  Eigen::VectorXf computeIoU_Eigen(const BoundingBox &box, const Eigen::MatrixXf &boxes)
  {
    Eigen::VectorXf x1 = boxes.col(0).cwiseMax(box.x_min);
    Eigen::VectorXf y1 = boxes.col(1).cwiseMax(box.y_min);
    Eigen::VectorXf x2 = boxes.col(2).cwiseMin(box.x_max);
    Eigen::VectorXf y2 = boxes.col(3).cwiseMin(box.y_max);

    Eigen::VectorXf intersection
      = ((x2 - x1).cwiseMax(0.0f)).array() * ((y2 - y1).cwiseMax(0.0f)).array();

    Eigen::VectorXf area1
      = (boxes.col(2) - boxes.col(0)).array() * (boxes.col(3) - boxes.col(1)).array();
    float area2
      = (box.x_max - box.x_min) * (box.y_max - box.y_min); // Fixed area calculation

    return intersection.array() / (area1.array() + area2 - intersection.array());
  }
  // Fast Non-Maximum Suppression (NMS) using Eigen
  std::vector<BoundingBox>
  fast_non_max_suppression(std::vector<BoundingBox> &bboxes, float iou_threshold)
  {
    if(bboxes.empty())
      return {};

    // Sort by confidence (descending)
    std::sort(bboxes.begin(), bboxes.end(),
              [](const BoundingBox &a, const BoundingBox &b) {
                return a.confidence > b.confidence;
              });

    int num_boxes = static_cast<int>(bboxes.size());
    Eigen::MatrixXf box_matrix(num_boxes, 4);
    Eigen::VectorXf confidences(num_boxes);
    std::vector<bool> keep(num_boxes, true);

    for(int i = 0; i < num_boxes; i++)
    {
      box_matrix(i, 0) = bboxes[i].x_min;
      box_matrix(i, 1) = bboxes[i].y_min;
      box_matrix(i, 2) = bboxes[i].x_max;
      box_matrix(i, 3) = bboxes[i].y_max;
      confidences(i) = bboxes[i].confidence;
    }

    std::vector<BoundingBox> final_bboxes;
    for(int i = 0; i < num_boxes; i++)
    {
      if(!keep[i])
        continue;
      final_bboxes.push_back(bboxes[i]);

      Eigen::VectorXf ious
        = computeIoU_Eigen(bboxes[i], box_matrix.bottomRows(num_boxes - i - 1));
      for(int j = 0; j < ious.size(); j++)
      {
        if(ious(j) > iou_threshold)
        {
          keep[i + j + 1] = false;
        }
      }
    }

    return final_bboxes;
  }

  void draw_bboxes(cv::Mat &image, const std::vector<BoundingBox> &bboxes)
  {
    for(const auto &box : bboxes)
    {
      cv::Rect rect(box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min);
      cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
      std::string label
        = objectClassToString(box.label) + " (" + std::to_string(box.confidence) + ")";
      cv::putText(image, label, cv::Point(box.x_min, box.y_min - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
  }

  void denormalizeAndScaleBoundingBox(std::vector<BoundingBox> &bboxes, int orig_w,
                                      int orig_h, int resize)
  {
    float scale_x = static_cast<float>(orig_w) / resize;
    float scale_y = static_cast<float>(orig_h) / resize;

    for(auto &box : bboxes)
    {
      box.x_min = static_cast<int>(box.x_min * resize * scale_x);
      box.y_min = static_cast<int>(box.y_min * resize * scale_y);
      box.x_max = static_cast<int>(box.x_max * resize * scale_x);
      box.y_max = static_cast<int>(box.y_max * resize * scale_y);
    }
  }

  Eigen::Matrix3d setIntrinsicMatrix(double fx, double fy, double cx, double cy)
  {
    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    return K;
  }

  Eigen::Matrix3d computeKInverse(const Eigen::Matrix3d &K) { return K.inverse(); }

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

    return ObjectClass::UNKNOWN;
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

    return "Unknown";
  }
}
