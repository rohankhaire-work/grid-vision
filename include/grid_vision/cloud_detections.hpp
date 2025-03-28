#ifndef CLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_
#define CLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_

#include "grid_vision/object_detection.hpp"

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <geometry_msgs/msg/point.hpp>

#include <vector>

namespace cloud_detections
{
  void buildKDTree(pcl::KdTreeFLANN<pcl::PointXYZ> &, pcl::PointCloud<pcl::PointXYZ>::Ptr,
                   const pcl::PointCloud<pcl::PointXYZI>::Ptr, const Eigen::Matrix3d &);

  std::vector<float>
  computeDepthForBoundingBoxes(pcl::KdTreeFLANN<pcl::PointXYZ> &,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr,
                               const std::vector<BoundingBox> &, uint16_t);

  geometry_msgs::msg::Point
  pixelTo3D(const cv::Point2f &, float, const Eigen::Matrix3d &);

}

#endif // CCLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_
