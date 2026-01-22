# ros2-aerial-manipulator
Autonomous manipulation system using ROS 2 Humble &amp; OpenManipulator-X. Features Visual Servoing, ArUco detection, and TF2 coordinate transformation.

## ArUco detection (Python)
Run the bundled Python ArUco detector (publishes `aruco_markers_msgs/MarkerArray` and an annotated image topic):

```bash
ros2 run omx_moveit_pick_cpp aruco_markers_node.py --ros-args \
  -p use_sim_time:=true \
  -p image_topic:=/camera/depth_camera/image_raw \
  -p camera_info_topic:=/camera/depth_camera/camera_info \
  -p camera_frame:=camera_optical_frame \
  -p dictionary:=DICT_5X5_100 \
  -p marker_size:=0.06 \
  -p annotated_image_topic:=/aruco/markers/image
```
