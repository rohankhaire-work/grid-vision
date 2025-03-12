cmake_minimum_required(VERSION 3.15)
project(grid_vision)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(spdlog REQUIRED)


add_executable(grid_vision src/grid_vision_node.cpp
                           src/object_detection.cpp)
target_include_directories(grid_vision PUBLIC include
                                               ${onnxruntime_INCLUDE_DIRS}/onnxruntime
                                               ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(grid_vision rclcpp cv_bridge image_transport sensor_msgs)

target_link_libraries(grid_vision
    ${onnxruntime_LIBRARY}
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
