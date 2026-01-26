#!/usr/bin/env python3
# 파일 위치: src/omx_moveit_pick_cpp/scripts/aruco_markers_node.py

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from aruco_markers_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
import tf_transformations

class ArucoMarkerNode(Node):
    def __init__(self):
        super().__init__('aruco_markers_node')

        self.declare_parameter("dictionary", "DICT_5X5_100")
        self.declare_parameter("marker_size", 0.06)
        self.declare_parameter("image_topic", "/camera/depth_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth_camera/camera_info")
        self.declare_parameter("annotated_image_topic", "/aruco/markers/image")
        
        dict_str = self.get_parameter("dictionary").value
        self.marker_size = self.get_parameter("marker_size").value
        image_topic = self.get_parameter("image_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        debug_topic = self.get_parameter("annotated_image_topic").value

        self.aruco_dict = self.get_aruco_dictionary(dict_str)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.camera_matrix = None
        self.dist_coeffs = None

        self.marker_pub = self.create_publisher(MarkerArray, '/aruco/markers', 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)

        self.get_logger().info(f"🚀 Aruco Node Ready! (No CvBridge) Topic: {image_topic}")

    def get_aruco_dictionary(self, dict_name):
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(dict_name, cv2.aruco.DICT_5X5_100))

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("✅ Camera Info Received!")

    def image_callback(self, msg):
        if self.camera_matrix is None: return

        try:
            dtype = np.uint8
            n_channels = 3
            if msg.encoding == 'rgb8' or msg.encoding == 'bgr8': n_channels = 3
            elif msg.encoding == 'mono8': n_channels = 1
            
            img_buf = np.frombuffer(msg.data, dtype=dtype)
            cv_image = img_buf.reshape(msg.height, msg.width, n_channels)
            if msg.encoding == 'rgb8': cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}"); return

        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            marker_array = MarkerArray()
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(cv_image, corners)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.05)
                
                marker_msg = Marker()
                marker_msg.header = msg.header
                marker_msg.id = int(ids[i][0])
                marker_msg.pose.pose.position.x = tvecs[i][0][0]
                marker_msg.pose.pose.position.y = tvecs[i][0][1]
                marker_msg.pose.pose.position.z = tvecs[i][0][2]
                
                rmat = cv2.Rodrigues(rvecs[i])[0]
                quat = tf_transformations.quaternion_from_matrix(np.vstack((np.hstack((rmat, [[0],[0],[0]])), [0,0,0,1])))
                marker_msg.pose.pose.orientation.x = quat[0]
                marker_msg.pose.pose.orientation.y = quat[1]
                marker_msg.pose.pose.orientation.z = quat[2]
                marker_msg.pose.pose.orientation.w = quat[3]
                marker_array.markers.append(marker_msg)

            self.marker_pub.publish(marker_array)

        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = cv_image.shape[0]; out_msg.width = cv_image.shape[1]
        out_msg.encoding = "bgr8"; out_msg.step = cv_image.shape[1] * 3
        out_msg.data = cv_image.tobytes()
        self.debug_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()