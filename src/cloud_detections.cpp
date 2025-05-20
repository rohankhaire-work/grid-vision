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
    seg.setDistanceThreshold(0.02);

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
      pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
      sor.setInputCloud(cloud_ptr);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1.0);
      sor.filter(filtered_cloud);

      LShapePose result;

      // Calculate centroid for height in 3D
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(filtered_cloud, centroid);

      pcl::PointXYZ center;
      center.x = centroid[0];
      center.y = centroid[1];
      center.z = centroid[2];

      geometry_msgs::msg::Pose pose;
      // Convert points to cv::Mat (Nx2)
      cv::Mat data_pts(filtered_cloud.points.size(), 2, CV_64F);
      for(size_t i = 0; i < filtered_cloud.points.size(); ++i)
      {
        data_pts.at<double>(i, 0) = static_cast<double>(filtered_cloud.points[i].z);
        data_pts.at<double>(i, 1) = static_cast<double>(filtered_cloud.points[i].x);
      }

      // Run PCA
      cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

      // Center of rectangle
      double cz = pca.mean.at<double>(0, 0);
      double cx = pca.mean.at<double>(0, 1);

      // Eigenvectors (unit vectors)
      cv::Vec2d major_axis(pca.eigenvectors.at<double>(0, 0),
                           pca.eigenvectors.at<double>(0, 1)); // principal direction
      cv::Vec2d minor_axis(pca.eigenvectors.at<double>(1, 0),
                           pca.eigenvectors.at<double>(1, 1)); // orthogonal direction

      std::vector<double> projections_major;
      std::vector<double> projections_minor;

      for(const auto &pt : filtered_cloud.points)
      {
        cv::Vec2d vec(pt.x - cx, pt.z - cz);
        projections_major.push_back(vec.dot(major_axis));
        projections_minor.push_back(vec.dot(minor_axis));
      }

      std::sort(projections_major.begin(), projections_major.end());
      std::sort(projections_minor.begin(), projections_minor.end());

      int n = projections_major.size();
      double min_major = projections_major[n * 0.05];
      double max_major = projections_major[n * 0.95];
      double min_minor = projections_minor[n * 0.05];
      double max_minor = projections_minor[n * 0.95];

      result.length = std::clamp(max_major - min_major, 0.1, 5.0); // 0.1 to 5 meters
      result.width = std::clamp(max_minor - min_minor, 0.1, 3.0);

      // Compute orientation
      double angle = atan2(major_axis[1], major_axis[0]);

      // Fill pose
      result.pose.position.x = cx;
      result.pose.position.y = center.y;
      result.pose.position.z = cz;

      tf2::Quaternion q;
      q.setRPY(0, -angle, 0); // rotation in XZ plane (around Y)
      q.normalize();
      result.pose.orientation = tf2::toMsg(q);

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
      if(!pcl::isFinite(pt) || pt.x <= 0.001f)
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
