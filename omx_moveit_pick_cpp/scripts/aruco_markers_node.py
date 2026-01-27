#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from aruco_markers_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
import math

def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

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
        self.first_log = False

        self.marker_pub = self.create_publisher(MarkerArray, '/aruco/markers', 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)

        self.get_logger().info(f"🚀 Aruco Node V3 (No-Crash Mode) on: {image_topic}")

    def get_aruco_dictionary(self, dict_name):
        ARUCO_DICT = {
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(dict_name, cv2.aruco.DICT_5X5_100))

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("✅ Camera Info Received! Ready to process images.")

    def image_callback(self, msg):
        if self.camera_matrix is None: return

        try:
            # 1. 디버깅용: 인코딩 확인 (터지기 직전에 로그 남기기)
            if not self.first_log:
                # self.get_logger().info(f"Processing Image: {msg.width}x{msg.height}, {msg.encoding}")
                self.first_log = True

            cv_image = None
            np_arr = np.frombuffer(msg.data, np.uint8)

            # 2. 이미지 변환 & 메모리 안전장치
            if msg.encoding == 'rgb8':
                cv_image = np_arr.reshape((msg.height, msg.width, 3))
                # [중요] 메모리를 강제로 연속적으로 재배치 (OpenCV 충돌 방지 핵심)
                cv_image = np.ascontiguousarray(cv_image) 
            elif msg.encoding == 'bgr8':
                cv_image = np_arr.reshape((msg.height, msg.width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                cv_image = np.ascontiguousarray(cv_image)
            else:
                return # 모르는 인코딩은 조용히 무시

            # 3. 마커 탐지
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                marker_array = MarkerArray()
                # 4. 좌표 계산 (여기가 충돌 위험 구간 -> 메모리 정리로 해결)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

                for i in range(len(ids)):
                    cv2.aruco.drawDetectedMarkers(cv_image, corners)
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.05)

                    marker_msg = Marker()
                    marker_msg.header = msg.header
                    marker_msg.id = int(ids[i][0])
                    
                    marker_msg.pose.pose.position.x = tvecs[i][0][0]
                    marker_msg.pose.pose.position.y = tvecs[i][0][1]
                    marker_msg.pose.pose.position.z = tvecs[i][0][2]

                    # tf_transformations 대체 함수 사용
                    rmat = cv2.Rodrigues(rvecs[i])[0]
                    quat = rotation_matrix_to_quaternion(rmat)
                    
                    marker_msg.pose.pose.orientation.x = quat[0]
                    marker_msg.pose.pose.orientation.y = quat[1]
                    marker_msg.pose.pose.orientation.z = quat[2]
                    marker_msg.pose.pose.orientation.w = quat[3]
                    marker_array.markers.append(marker_msg)

                self.marker_pub.publish(marker_array)

            # 5. 디버그 이미지 발행
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = cv_image.shape[0]
            out_msg.width = cv_image.shape[1]
            out_msg.encoding = 'rgb8'
            out_msg.step = cv_image.shape[1] * 3
            out_msg.data = cv_image.tobytes()
            self.debug_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Critical Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()