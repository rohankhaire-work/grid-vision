#include "grid_vision/occupancy_grid.hpp"

OccupancyGridMap::OccupancyGridMap(const std::string &base_link, uint8_t grid_x,
                                   uint8_t grid_y, double resolution)
{
  // Initialize grid map
  grid_map_ = grid_map::GridMap({"log_odds", "occupancy"});
  grid_map_.setFrameId(base_link);
  grid_map_.setGeometry(grid_map::Length(grid_x, grid_y), resolution);
  grid_map_.setPosition(grid_map::Position(grid_x / 3, 0.0));
  grid_map_["log_odds"].setConstant(log_odds_prior_);
  grid_map_["occupancy"].setConstant(init_probability_);
}

void OccupancyGridMap::updateMap(grid_map::GridMap &grid_map)
{
  // Decay all cells
  grid_map["log_odds"].array() += log_odds_decay_;

  grid_map["log_odds"]
    = grid_map["log_odds"].cwiseMax(min_log_odds_).cwiseMin(max_log_odds_);

  // Convert log-odds to probability
  for(grid_map::GridMapIterator it(grid_map); !it.isPastEnd(); ++it)
  {
    float log_odds_value = grid_map.at("log_odds", *it);
    float probability = 1.0f / (1.0f + std::exp(-log_odds_value));
    grid_map.at("occupancy", *it) = probability;
  }
}

void OccupancyGridMap::updateMap(grid_map::GridMap &grid_map,
                                 const std::vector<geometry_msgs::msg::Point> &base_points,
                                 const std::vector<BoundingBox> &bboxes)
{
  // Decay all cells
  grid_map["log_odds"].array() += log_odds_decay_;

  // Process detected objects & update occupied cells
  for(size_t idx = 0; idx < base_points.size(); ++idx)
  {
    auto base_point = base_points[idx];
    auto bbox_width = bboxes[idx].x_max - bboxes[idx].x_min;
    auto bbox_label = bboxes[idx].label;
    if(bbox_label == ObjectClass::BIKE || bbox_label == ObjectClass::MOTORBIKE
       || bbox_label == ObjectClass::PERSON || bbox_label == ObjectClass::VEHICLE)
    {
      // Compute the Rectangel from center point
      std::array<geometry_msgs::msg::Point, 4> occ_corners
        = computeBoundingBox3D(base_point, bbox_label);

      // Update the grid cells in the map
      updateGridCellsFast(grid_map, occ_corners);
    }
  }
  grid_map["log_odds"]
    = grid_map["log_odds"].cwiseMax(min_log_odds_).cwiseMin(max_log_odds_);
  // Convert log-odds to probability
  for(grid_map::GridMapIterator it(grid_map); !it.isPastEnd(); ++it)
  {
    float log_odds_value = grid_map.at("log_odds", *it);
    float probability = 1.0f / (1.0f + std::exp(-log_odds_value));
    grid_map.at("occupancy", *it) = probability;
  }
}

std::array<geometry_msgs::msg::Point, 4>
OccupancyGridMap::computeBoundingBox3D(const geometry_msgs::msg::Point &base_center,
                                       ObjectClass label)
{
  std::array<geometry_msgs::msg::Point, 4> corners;

  // Define 4 corner points in the base frame
  geometry_msgs::msg::Point p1, p2, p3, p4;
  auto estimated_depth = getEstimatedDepth(label);

  // Left-Front
  corners[0].x = base_center.x + estimated_depth;
  corners[0].y = base_center.y + (estimated_depth / 2);
  corners[0].z = base_center.z;

  // Right-Front
  corners[1].x = base_center.x + estimated_depth;
  corners[1].y = base_center.y - (estimated_depth / 2);
  corners[1].z = base_center.z;

  // Right-Back (At base center)
  corners[2].x = base_center.x;
  corners[2].y = base_center.y - (estimated_depth / 2);
  corners[2].z = base_center.z;

  // Left-Back (At base center)
  corners[3].x = base_center.x;
  corners[3].y = base_center.y + (estimated_depth / 2);
  corners[3].z = base_center.z;

  return corners;
}

void OccupancyGridMap::updateGridCellsFast(
  grid_map::GridMap &grid_map,
  const std::array<geometry_msgs::msg::Point, 4> &bbox_corners)
{
  // Convert bbox points to grid coordinates
  grid_map::Index min_index, max_index;
  bool valid = true;

  for(size_t i = 0; i < 4; i++)
  {
    grid_map::Position pos(bbox_corners[i].x, bbox_corners[i].y);
    grid_map::Index index;
    if(!grid_map.getIndex(pos, index))
    {
      valid = false;
      break;
    }

    if(i == 0)
    {
      min_index = max_index = index;
    }
    else
    {
      min_index.x() = std::min(min_index.x(), index.x());
      min_index.y() = std::min(min_index.y(), index.y());
      max_index.x() = std::max(max_index.x(), index.x());
      max_index.y() = std::max(max_index.y(), index.y());
    }
  }

  if(!valid)
    return; // Skip invalid bounding boxes

  // Access the grid layer directly
  Eigen::MatrixXf &grid_data = grid_map["log_odds"];

  // Update the entire block in one operation
  grid_data
    .block(min_index.x(), min_index.y(), max_index.x() - min_index.x() + 1,
           max_index.y() - min_index.y() + 1)
    .array()
    += 0.85f;
}

float OccupancyGridMap::getEstimatedDepth(ObjectClass class_label)
{
  switch(class_label)
  {
  case ObjectClass::VEHICLE: return 4.5f;   // Car (4.5m depth)
  case ObjectClass::PERSON: return 0.6f;    // Pedestrian (1.2m depth)
  case ObjectClass::BIKE: return 2.5f;      // Bicycle (2.5m depth)
  case ObjectClass::MOTORBIKE: return 2.5f; // Generic object (3.0m depth)
  }

  return -1.0f;
}
