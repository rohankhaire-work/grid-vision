#ifndef OCCUPANCY_GRID__OCCUPANCY_GRID_HPP_
#define OCCUPANCY_GRID__OCCUPANCY_GRID_HPP_

#include "grid_vision/object_detection.hpp"

#include <grid_map_ros/grid_map_ros.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <vector>
#include <array>
#include <string>

class OccupancyGridMap
{
public:
  OccupancyGridMap(const std::string &, uint8_t, uint8_t, double);
  void updateMap(grid_map::GridMap &, const std::vector<geometry_msgs::msg::Point> &,
                 const std::vector<BoundingBox> &);

  grid_map::GridMap grid_map_;

private:
  float log_odds_free_ = -0.4f;
  float log_odds_occupied_ = 0.85f;
  float log_odds_prior_ = 0.0f;
  float init_probability_ = 0.5f;
  float log_odds_decay_ = -0.05f;

  std::array<geometry_msgs::msg::Point, 4>
  computeBoundingBox3D(const geometry_msgs::msg::Point &, float, ObjectClass);

  void updateGridCellsFast(grid_map::GridMap &,
                           const std::array<geometry_msgs::msg::Point, 4> &);

  float getEstimatedDepth(ObjectClass);
};

#endif //  OCCUPANCY_GRID__OCCUPANCY_GRID_HPP_
