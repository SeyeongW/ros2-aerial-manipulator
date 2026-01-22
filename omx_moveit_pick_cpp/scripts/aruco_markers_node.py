#!/usr/bin/env python3
import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from aruco_markers_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge


DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


class ArucoMarkersNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_markers_node")

        self.declare_parameter("image_topic", "/camera/depth_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth_camera/camera_info")
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.declare_parameter("dictionary", "DICT_5X5_100")
        self.declare_parameter("marker_size", 0.06)
        self.declare_parameter("markers_topic", "/aruco/markers")
        self.declare_parameter("annotated_image_topic", "/aruco/markers/image")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        dictionary_name = self.get_parameter("dictionary").get_parameter_value().string_value
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        markers_topic = self.get_parameter("markers_topic").get_parameter_value().string_value
        annotated_topic = self.get_parameter("annotated_image_topic").get_parameter_value().string_value

        dict_id = DICT_MAP.get(dictionary_name)
        if dict_id is None:
            raise ValueError(f"Unknown dictionary '{dictionary_name}'")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector: Optional[cv2.aruco.ArucoDetector] = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self.marker_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.image_pub = self.create_publisher(Image, annotated_topic, 10)

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, image_topic, self.image_cb, 10)

        self.get_logger().info(
            f"Aruco node ready. image={image_topic} camera_info={camera_info_topic} "
            f"markers={markers_topic} annotated={annotated_topic} dict={dictionary_name}"
        )

    def camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def detect_markers(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

    def image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detect_markers(gray)
        marker_array = MarkerArray()
        marker_array.header.stamp = msg.header.stamp
        marker_array.header.frame_id = self.camera_frame

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            for idx, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[idx].reshape((3, 1))
                tvec = tvecs[idx].reshape((3, 1))
                rot, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = rotation_matrix_to_quaternion(rot)

                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = self.camera_frame
                pose.pose.position.x = float(tvec[0])
                pose.pose.position.y = float(tvec[1])
                pose.pose.position.z = float(tvec[2])
                pose.pose.orientation.x = float(qx)
                pose.pose.orientation.y = float(qy)
                pose.pose.orientation.z = float(qz)
                pose.pose.orientation.w = float(qw)

                marker = Marker()
                marker.id = int(marker_id)
                marker.pose = pose
                marker_array.markers.append(marker)

                cv2.drawFrameAxes(
                    cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size * 0.5
                )

            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        self.marker_pub.publish(marker_array)
        annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        annotated_msg.header.stamp = msg.header.stamp
        annotated_msg.header.frame_id = msg.header.frame_id
        self.image_pub.publish(annotated_msg)


def main() -> None:
    rclpy.init()
    node = ArucoMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
