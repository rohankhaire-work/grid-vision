#include "grid_vision/cloud_detections.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace cloud_detections
{
  void buildKDTree(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr image_points,
                   const pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_points,
                   const Eigen::Matrix3d &K)
  {
    for(const auto &p : lidar_points->points)
    {
      // Ignore points behind the camera
      if(p.z <= 0)
        continue;
      // Convert to homogeneous coordinates
      Eigen::Vector3d cam_point(p.x, p.y, p.z);
      Eigen::Vector3d img_point = K * cam_point; // K * [X Y Z]^T

      // Normalize to get pixel coordinates
      float u = img_point.x() / img_point.z();
      float v = img_point.y() / img_point.z();

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
      search_point.x = bbox.x_min + ((bbox.x_max - bbox.x_min) / 2.0f);
      search_point.y = bbox.y_min + ((bbox.y_max - bbox.y_min) / 2.0f);
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
  pixelTo3D(const cv::Point2f &pixel, float depth, const Eigen::Matrix3d &K_inv)
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

  pcl::PointCloud<pcl::PointXYZI>
  segmentGroundPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud)
  {
    // Object for plane segmentation
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // Configure the segmentation object
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.04);

    seg.setInputCloud(input_cloud);
    seg.segment(*inliers, *coefficients);

    if(inliers->indices.size() == 0)
    {
      PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
      return pcl::PointCloud<pcl::PointXYZI>();
    }

    // Extract points not belonging to the plane (ground)
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true); // true to remove the plane, false to keep the plane
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_no_ground(
      new pcl::PointCloud<pcl::PointXYZI>);
    extract.filter(*cloud_no_ground);

    return *cloud_no_ground;
  }

  void
  bboxPoseEstimation(const std::vector<pcl::PointCloud<pcl::PointXYZI>> &bboxes_cloud,
                     std::vector<LShapePose> &lshape_pose)
  {
    for(const auto &bbox_cloud : bboxes_cloud)
    {
      // Filter PointCloud
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
      *cloud_ptr = bbox_cloud;
      pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
      pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;
      outrem.setInputCloud(cloud_ptr);    // cloud is pcl::PointCloud<pcl::PointXYZ>::Ptr
      outrem.setRadiusSearch(0.4);        // radius in meters (example: 10 cm)
      outrem.setMinNeighborsInRadius(10); // minimum number of neighbors required
      outrem.filter(filtered_cloud);

      // Get cnetroid
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(filtered_cloud, centroid);

      pcl::PointXYZ center;
      center.x = centroid[0];
      center.y = centroid[1];
      center.z = centroid[2];

      LShapePose result;

      // Project points to XY
      std::vector<cv::Point2f> points_2d;

      for(const auto &pt : filtered_cloud.points)
      {
        points_2d.emplace_back(pt.z, pt.x);
      }

      if(points_2d.empty())
        continue;
      // Get 2D min area rectangle
      cv::RotatedRect rect = cv::minAreaRect(points_2d);

      // Extract angle and center
      float angle_deg = rect.angle;
      float width = rect.size.width;
      float length = rect.size.height;

      if(width > length)
      {
        std::swap(width, length);
        angle_deg += 90.0f;
      }

      float angle_rad = angle_deg * CV_PI / 180.0f;

      // Pose (position)
      result.pose.position.x = rect.center.y;
      result.pose.position.y = center.y;
      result.pose.position.z = rect.center.x;

      // Pose (orientation as quaternion)
      tf2::Quaternion q;
      q.setRPY(0, -angle_rad, 0); // Rotation around Z only
      result.pose.orientation.x = q.x();
      result.pose.orientation.y = q.y();
      result.pose.orientation.z = q.z();
      result.pose.orientation.w = q.w();

      // Dimensions
      result.length = length;
      result.width = width;

      lshape_pose.emplace_back(result);
    }
  }

  // Projects 3D cloud into image space and collects 3D points inside each bbox
  void
  extractCloudPerBBox(const pcl::PointCloud<pcl::PointXYZI> &cloud,
                      const Eigen::Matrix3d &K, const std::vector<BoundingBox> &bboxes,
                      std::vector<pcl::PointCloud<pcl::PointXYZI>> &output_clouds,
                      int image_width, int image_height)
  {
    output_clouds.clear();

    // Set the size
    output_clouds.resize(bboxes.size());

    // For each 3D point
    for(const auto &pt : cloud.points)
    {
      if(!pcl::isFinite(pt) || pt.z <= 0.001f)
        continue;

      // Convert to homogeneous coordinates
      Eigen::Vector3d cam_point(pt.x, pt.y, pt.z);
      Eigen::Vector3d img_point = K * cam_point; // K * [X Y Z]^T

      // Normalize to get pixel coordinates
      float u = img_point.x() / img_point.z();
      float v = img_point.y() / img_point.z();

      // Skip points that project outside image
      if(u < 0 || u >= image_width || v < 0 || v >= image_height)
        continue;

      // Check which bbox (if any) contains the projected pixel
      for(size_t i = 0; i < bboxes.size(); ++i)
      {
        if(u >= bboxes[i].x_min && u <= bboxes[i].x_max && v >= bboxes[i].y_min
           && v <= bboxes[i].y_max)
        {
          output_clouds[i].points.push_back(pt);
          break;
        }
      }
    }

    // Set cloud sizes
    for(auto &c : output_clouds)
    {
      c.width = c.points.size();
      c.height = 1;
      c.is_dense = true;
    }
  }

  std::vector<LShapePose>
  computeBBoxPose(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                  const Eigen::Matrix3d &K, const std::vector<BoundingBox> &bboxes,
                  int image_height, int image_width)
  {
    // Segment the ground plane
    pcl::PointCloud<pcl::PointXYZI> segmented_cloud = segmentGroundPlane(input_cloud);

    if(segmented_cloud.empty())
      return {};

    // Find the clouds for each bounding boxes
    std::vector<pcl::PointCloud<pcl::PointXYZI>> output_clouds;
    extractCloudPerBBox(segmented_cloud, K, bboxes, output_clouds, image_width,
                        image_height);

    // Get the Pose & dims of each BBox using PCA
    std::vector<LShapePose> bboxes_pose;
    bboxPoseEstimation(output_clouds, bboxes_pose);

    return bboxes_pose;
  }
}
