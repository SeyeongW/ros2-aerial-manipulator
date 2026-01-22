from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_frame = LaunchConfiguration("camera_frame")
    dictionary = LaunchConfiguration("dictionary")
    marker_size = LaunchConfiguration("marker_size")
    annotated_image_topic = LaunchConfiguration("annotated_image_topic")
    markers_topic = LaunchConfiguration("markers_topic")

    target_id = LaunchConfiguration("target_id")
    base_frame = LaunchConfiguration("base_frame")
    planning_group = LaunchConfiguration("planning_group")
    ee_link = LaunchConfiguration("ee_link")
    do_place = LaunchConfiguration("do_place")
    place_x = LaunchConfiguration("place_x")
    place_y = LaunchConfiguration("place_y")
    place_z = LaunchConfiguration("place_z")
    place_roll = LaunchConfiguration("place_roll")
    place_pitch = LaunchConfiguration("place_pitch")
    place_yaw = LaunchConfiguration("place_yaw")

    aruco_node = Node(
        package="omx_moveit_pick_cpp",
        executable="aruco_markers_node.py",
        name="aruco_markers_node",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "camera_frame": camera_frame,
                "dictionary": dictionary,
                "marker_size": marker_size,
                "markers_topic": markers_topic,
                "annotated_image_topic": annotated_image_topic,
            }
        ],
    )

    pick_place_node = Node(
        package="omx_moveit_pick_cpp",
        executable="omx_moveit_pick_node",
        name="omx_moveit_pick_node",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "markers_topic": markers_topic,
                "target_id": target_id,
                "base_frame": base_frame,
                "planning_group": planning_group,
                "ee_link": ee_link,
                "do_place": do_place,
                "place_x": place_x,
                "place_y": place_y,
                "place_z": place_z,
                "place_roll": place_roll,
                "place_pitch": place_pitch,
                "place_yaw": place_yaw,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("image_topic", default_value="/camera/depth_camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/depth_camera/camera_info"),
            DeclareLaunchArgument("camera_frame", default_value="camera_optical_frame"),
            DeclareLaunchArgument("dictionary", default_value="DICT_5X5_100"),
            DeclareLaunchArgument("marker_size", default_value="0.06"),
            DeclareLaunchArgument("markers_topic", default_value="/aruco/markers"),
            DeclareLaunchArgument("annotated_image_topic", default_value="/aruco/markers/image"),
            DeclareLaunchArgument("target_id", default_value="0"),
            DeclareLaunchArgument("base_frame", default_value="link1"),
            DeclareLaunchArgument("planning_group", default_value="arm"),
            DeclareLaunchArgument("ee_link", default_value="end_effector_link"),
            DeclareLaunchArgument("do_place", default_value="true"),
            DeclareLaunchArgument("place_x", default_value="0.20"),
            DeclareLaunchArgument("place_y", default_value="0.00"),
            DeclareLaunchArgument("place_z", default_value="0.10"),
            DeclareLaunchArgument("place_roll", default_value="0.0"),
            DeclareLaunchArgument("place_pitch", default_value="0.0"),
            DeclareLaunchArgument("place_yaw", default_value="0.0"),
            aruco_node,
            pick_place_node,
        ]
    )
