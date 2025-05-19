#ifndef CLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_
#define CLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_

#include "grid_vision/object_detection.hpp"

#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <geometry_msgs/msg/pose.hpp>

#include <vector>

struct LShapePose
{
  geometry_msgs::msg::Pose pose;
  double length; // along principal axis
  double width;  // perpendicular to principal axis
};

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

  pcl::PointCloud<pcl::PointXYZ>
  segmentGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &);

  void bboxPoseEstimation(const std::vector<pcl::PointCloud<pcl::PointXYZ>> &,
                          std::vector<LShapePose> &);

  void extractCloudPerBBox(const pcl::PointCloud<pcl::PointXYZ> &,
                           const Eigen::Matrix3d &, const std::vector<BoundingBox> &,
                           std::vector<pcl::PointCloud<pcl::PointXYZ>> &, int, int);

  std::vector<LShapePose>
  computeBBoxPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const std::Matrix3d &,
                  const std::vector<BoundingBox> &, int, int);

}

#endif // CCLOUD_DETECTIONS__CLOUD_DETECTIONS_HPP_
