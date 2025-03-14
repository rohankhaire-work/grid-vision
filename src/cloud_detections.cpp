#include "grid_vision/cloud_detections.hpp"
#include <geometry_msgs/msg/detail/point__struct.hpp>

namespace cloud_detections
{
  void
  buildKDTree(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree,
              pcl::PointCloud<pcl::PointXYZ>::Ptr image_points,
              const pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_points, const cv::Mat &K)
  {
    for(const auto &p : lidar_points->points)
    {
      // Ignore points behind the camera
      if(p.x <= 0)
        continue;

      // Convert to homogeneous coordinates
      cv::Mat cam_point = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
      cv::Mat img_point = K * cam_point; // K * [X Y Z]^T

      // Normalize to get pixel coordinates
      float u = img_point.at<double>(0, 0) / img_point.at<double>(2, 0);
      float v = img_point.at<double>(1, 0) / img_point.at<double>(2, 0);

      // Store in pcl::PointXYZ (u → x, v → y, depth → z)
      pcl::PointXYZ pt;
      pt.x = u;   // u (image coordinate)
      pt.y = v;   // v (image coordinate)
      pt.z = p.z; // Depth value

      image_points->push_back(pt);
    }

    // Ensure KD-Tree input cloud is set properly
    if(!image_points->empty())
    {
      kdtree.setInputCloud(image_points);
    }
  }

  // Compute depth for each bounding box using k=10 nearest neighbors
  std::vector<float>
  computeDepthForBoundingBoxes(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr image_points,
                               const std::vector<BoundingBox> &bboxes, uint16_t k = 10)
  {
    size_t num_bboxes = bboxes.size();
    std::vector<float> depths(num_bboxes, -1.0f); // Preallocate depth storage

    for(size_t i = 0; i < num_bboxes; ++i)
    {
      const auto &bbox = bboxes[i];

      // Compute bounding box center
      pcl::PointXYZ search_point;
      search_point.x = (bbox.x_max - bbox.x_min) / 2.0f;
      search_point.y = (bbox.y_max - bbox.y_min) / 2.0f;
      search_point.z = 0.0f;

      std::vector<int> point_indices(k);
      std::vector<float> point_distances(k);

      if(kdtree.nearestKSearch(search_point, k, point_indices, point_distances) > 0)
      {
        std::vector<float> depth_values;
        for(int idx : point_indices)
        {
          if(idx >= 0 && idx < image_points->size())
          {
            depth_values.push_back(image_points->points[idx].z);
          }
        }

        if(!depth_values.empty())
        {
          // Use the **median** depth for stability
          size_t mid = depth_values.size() / 2;
          std::nth_element(depth_values.begin(), depth_values.begin() + mid,
                           depth_values.end());
          depths[i] = depth_values[mid];
        }
      }
    }

    return depths;
  }

  geometry_msgs::msg::Point
  pixelTo3D(const cv::Point2f &pixel, float depth, const cv::Mat &K_inv)
  {
    // Convert pixel coordinates to homogeneous coordinates
    cv::Mat pixel_homogeneous = (cv::Mat_<double>(3, 1) << pixel.x, pixel.y, 1.0);
    // Compute the 3D point in camera frame: X_cam = K_inv * (u, v, 1) * depth
    cv::Mat cam_point_mat = K_inv * pixel_homogeneous * depth;

    geometry_msgs::msg::Point cam_point;
    cam_point.x = cam_point_mat.at<float>(0, 0);
    cam_point.y = cam_point_mat.at<float>(1, 0);
    cam_point.z = cam_point_mat.at<float>(2, 0);

    return cam_point;
  }
}
