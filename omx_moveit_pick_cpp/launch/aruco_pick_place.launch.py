# 파일 위치: src/omx_moveit_pick_cpp/launch/aruco_pick_place.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time")
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    dictionary = LaunchConfiguration("dictionary")
    marker_size = LaunchConfiguration("marker_size")
    annotated_image_topic = LaunchConfiguration("annotated_image_topic")
    enable_pose = LaunchConfiguration("enable_pose")
    publish_debug = LaunchConfiguration("publish_debug")

    # ArUco node (Python)
    aruco_node = Node(
        package="omx_moveit_pick_cpp",
        executable="aruco_markers_node.py",
        name="aruco_markers_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "dictionary": dictionary,
                "marker_size": marker_size,
                "annotated_image_topic": annotated_image_topic,
                "enable_pose": enable_pose,          
                "publish_debug": publish_debug,      
            }
        ],
    )

    # Picker node (C++)
    pick_place_node = Node(
        package="omx_moveit_pick_cpp",
        executable="omx_moveit_pick_node",
        name="omx_moveit_pick_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("image_topic", default_value="/camera/depth_camera/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/depth_camera/camera_info"),
            DeclareLaunchArgument("dictionary", default_value="DICT_5X5_100"),
            DeclareLaunchArgument("marker_size", default_value="0.06"),
            DeclareLaunchArgument("annotated_image_topic", default_value="/aruco/markers/image"),
            DeclareLaunchArgument("enable_pose", default_value="true"),     
            DeclareLaunchArgument("publish_debug", default_value="true"),   
            aruco_node,
            pick_place_node,
        ]
    )
