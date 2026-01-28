cmake_minimum_required(VERSION 3.8)
project(omx_moveit_pick_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(aruco_markers_msgs REQUIRED)
find_package(sensor_msgs REQUIRED) 

# ==========================================
# 1. 기존 노드 (omx_moveit_pick_node)
# ==========================================
add_executable(omx_moveit_pick_node src/omx_moveit_pick_node.cpp)
ament_target_dependencies(omx_moveit_pick_node
  rclcpp
  rclcpp_action
  control_msgs
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_geometry_msgs
  aruco_markers_msgs
  sensor_msgs
)

# ==========================================
# 2. [추가] 새로 만든 그립 노드 (omx_grip_node) 등록
# ==========================================
add_executable(omx_grip_node src/omx_grip_node.cpp)

ament_target_dependencies(omx_grip_node
  rclcpp
  rclcpp_action
  control_msgs
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  aruco_markers_msgs
)

# ==========================================
# 3. 설치 (Install)
# ==========================================
install(TARGETS
  omx_moveit_pick_node
  omx_grip_node
  DESTINATION lib/${PROJECT_NAME}
)

# ===== Python node (aruco detector) =====
install(PROGRAMS
  scripts/aruco_markers_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===== launch files =====
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()