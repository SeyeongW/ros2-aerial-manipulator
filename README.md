# ros2-aerial-manipulator
Autonomous manipulation system using ROS 2 Humble &amp; OpenManipulator-X. Features Visual Servoing, ArUco detection, and TF2 coordinate transformation.

## ArUco detection (Python)
Run the bundled Python ArUco detector (publishes `aruco_markers_msgs/MarkerArray` and an annotated image topic):

```bash
colcon build --packages-select omx_moveit_pick_cpp
source install/setup.bash
ros2 run omx_moveit_pick_cpp aruco_markers_node.py --ros-args \
  -p use_sim_time:=true \
  -p image_topic:=/camera/depth_camera/image_raw \
  -p camera_info_topic:=/camera/depth_camera/camera_info \
  -p camera_frame:=camera_optical_frame \
  -p dictionary:=DICT_5X5_100 \
  -p marker_size:=0.06 \
  -p annotated_image_topic:=/aruco/markers/image
```

## Pick & place node
Run the MoveIt pick-and-place node (listens on `/aruco/markers`):

```bash
source install/setup.bash
ros2 run omx_moveit_pick_cpp omx_moveit_pick_node --ros-args \
  -p target_id:=0 \
  -p base_frame:=link1 \
  -p planning_group:=arm \
  -p ee_link:=end_effector_link \
  -p do_place:=true \
  -p place_x:=0.20 \
  -p place_y:=0.00 \
  -p place_z:=0.10
```

## Launch (Aruco + Pick/Place)
Run both nodes together with a launch file:

```bash
source install/setup.bash
ros2 launch omx_moveit_pick_cpp aruco_pick_place.launch.py \
  use_sim_time:=true \
  image_topic:=/camera/depth_camera/image_raw \
  camera_info_topic:=/camera/depth_camera/camera_info \
  camera_frame:=camera_optical_frame \
  dictionary:=DICT_5X5_100 \
  marker_size:=0.06 \
  target_id:=0 \
  do_place:=true \
  place_x:=0.20 \
  place_y:=0.00 \
  place_z:=0.10
```
