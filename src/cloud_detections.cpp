#include "grid_vision/cloud_detections.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <geometry_msgs/msg/detail/point__struct.hpp>

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

  pcl::PointCloud<pcl::PointXYZ>
  segmentGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud)
  {
    // Object for plane segmentation
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // Configure the segmentation object
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);

    seg.setInputCloud(*input_cloud);
    seg.segment(*inliers, *coefficients);

    if(inliers->indices.size() == 0)
    {
      PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
      return -1;
    }

    // Extract points not belonging to the plane (ground)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(*input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true); // true to remove the plane, false to keep the plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(
      new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(cloud_no_ground);
  }

  void bboxPoseEstimation(const std::vector<pcl::PointCloud<pcl::PointXYZ>> &bbox_cloud,
                          std::vector<LShapePose> &lshape_pose)
  {
    for(const auto &bbox_cloud : bboxes_cloud)
    {
      LShapePose result;

      // Calculate centroid for height in 3D
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud, centroid);

      pcl::PointXYZ center;
      center.x = centroid[0];
      center.y = centroid[1];
      center.z = centroid[2];

      geometry_msgs::Pose pose;
      // Convert points to cv::Mat (Nx2)
      cv::Mat data_pts(bbox_cloud.points.size(), 2, CV_64F);
      for(size_t i = 0; i < bbox_cloud.points..size(); ++i)
      {
        data_pts.at<double>(i, 0) = static_cast<double>(xz_points[i].z); // X
        data_pts.at<double>(i, 1) = static_cast<double>(xz_points[i].x); // Y
      }

      // Run PCA
      cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);

      // Center of rectangle
      double cx = pca.mean.at<double>(0, 0);
      double cz = pca.mean.at<double>(0, 1);

      // Eigenvectors (unit vectors)
      cv::Vec2d major_axis(pca.eigenvectors.at<double>(0, 0),
                           pca.eigenvectors.at<double>(0, 1)); // principal direction
      cv::Vec2d minor_axis(pca.eigenvectors.at<double>(1, 0),
                           pca.eigenvectors.at<double>(1, 1)); // orthogonal direction

      // Project all points onto both axes to find extent
      double min_major = DBL_MAX, max_major = -DBL_MAX;
      double min_minor = DBL_MAX, max_minor = -DBL_MAX;

      for(const auto &pt : bbox_cloud.points)
      {
        cv::Vec2d vec(pt.z - cx, pt.x - cz); // centered point
        double proj_major = vec.dot(major_axis);
        double proj_minor = vec.dot(minor_axis);

        min_major = std::min(min_major, proj_major);
        max_major = std::max(max_major, proj_major);
        min_minor = std::min(min_minor, proj_minor);
        max_minor = std::max(max_minor, proj_minor);
      }

      result.length = max_major - min_major;
      result.width = max_minor - min_minor;

      // Compute orientation
      double angle = atan2(major_axis[1], major_axis[0]);

      // Fill pose
      geometry_msgs::Pose &pose = result.pose;
      pose.position.x = cz;
      pose.position.y = center.y;
      pose.position.z = cx;

      tf2::Quaternion q;
      q.setRPY(0, -angle, 0); // rotation in XZ plane (around Y)
      q.normalize();
      pose.orientation = tf2::toMsg(q);

      lshape_pose.emplace_back(result);
    }
  }

  // Projects 3D cloud into image space and collects 3D points inside each bbox
  void
  extractCloudPerBBox(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                      const Eigen::Matrix3d &K, const std::vector<BoundingBox> &bboxes,
                      std::vector<pcl::PointCloud<pcl::PointXYZ>> &output_clouds,
                      int image_width, int image_height)
  {
    output_clouds.clear();

    // Set the size
    output_clouds.resize(bbox_size);

    // Initialize output clouds
    for(auto &c : output_clouds)
    {
      c.reset(new pcl::PointCloud<pcl::PointXYZ>());
    }

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
          output_clouds[i]->points.push_back(pt);
          break;
        }
      }
    }

    // Set cloud sizes
    for(auto &c : output_clouds)
    {
      c->width = c->points.size();
      c->height = 1;
      c->is_dense = true;
    }
  }

  std::vector<LShapePose>
  computeBBoxPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                  const std::Matrix3d &K, const std::vector<BoundingBox> &bboxes,
                  int image_height, int image_width)
  {
    // Segment the ground plane
    pcl::PointCloud<pcl::PointXYZ> segmented_cloud = segmentGroundPlane(input_cloud);

    // Find the clouds for each bounding boxes
    std::vector<pcl::PointCloud<pcl::PointXYZ>> output_clouds;
    extractCloudPerBBox(segmented_cloud, K, bboxes, output_clouds, image_width,
                        image_height);

    // Get the Pose & dims of each BBox using PCA
    std::vector<LShapePose> bboxes_pose;
    bboxPoseEstimation(output_clouds, bboxes_pose);

    return bboxes_pose;
  }
}
