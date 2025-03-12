cmake_minimum_required(VERSION 3.15)
project(grid_vision)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

add_executable(grid_vision src/grid_vision_node.cpp)
target_include_directories(grid_vision PRIVATE include)
ament_target_dependencies(grid_vision rclcpp cv_bridge image_transport sensor_msgs)

install(
  DIRECTORY config
            launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
