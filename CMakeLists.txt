cmake_minimum_required(VERSION 3.15)
project(grid_vision)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(tensorrt REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(grid_map_ros REQUIRED)

add_executable(grid_vision src/grid_vision_node.cpp
                           src/object_detection.cpp
                           src/cloud_detections.cpp
                           src/occupancy_grid.cpp)
target_include_directories(grid_vision PRIVATE include
                                            ${onnxruntime_INCLUDE_DIRS}/onnxruntime
                                            ${OpenCV_INCLUDE_DIRS}
                                            ${PCL_INCLUDE_DIRS}
                                            ${CUDAToolkit_INCLUDE_DIR} 
                                            ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(grid_vision rclcpp cv_bridge image_transport pcl_conversions pcl_ros
                                      tf2_ros tf2_geometry_msgs grid_map_ros)

target_link_libraries(grid_vision
    ${onnxruntime_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    spdlog::spdlog_header_only
    Eigen3::Eigen
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer.so)

install(TARGETS grid_vision
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
