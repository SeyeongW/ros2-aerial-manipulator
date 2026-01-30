#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from aruco_markers_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

# 쿼터니언 변환 함수
def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = 0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        qw, qx, qy, qz = (R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        qw, qx, qy, qz = (R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        qw, qx, qy, qz = (R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S
    return [qx, qy, qz, qw]

class ArucoRealSense(Node):
    def __init__(self):
        super().__init__("aruco_realsense_node")
        
        # [RealSense 토픽 설정]
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("marker_size", 0.06) # 6cm (실측 후 수정 필수!)

        self.marker_size = self.get_parameter("marker_size").value
        
        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # 통신
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, "/aruco/markers", 10)
        
        # 이미지 구독
        img_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("info_topic").value

        self.create_subscription(CameraInfo, info_topic, self.info_cb, 10)
        self.create_subscription(Image, img_topic, self.img_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, 10)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_depth = None

        self.get_logger().info("RealSense ArUco Node Started.")

    def info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3,3))
            self.dist_coeffs = np.array(msg.d)

    def depth_cb(self, msg):
        # 깊이 이미지 변환 (mm -> m)
        dtype = np.uint16 # RealSense Depth는 uint16 (mm)
        data = np.frombuffer(msg.data, dtype=dtype)
        if len(data) == msg.width * msg.height:
            self.latest_depth = data.reshape((msg.height, msg.width)).astype(np.float32) * 0.001

    def img_cb(self, msg):
        if self.camera_matrix is None: return
        
        # 이미지 변환 (YUV/RGB -> Gray)
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = buf.reshape((msg.height, msg.width, 3)) # RGB8 가정
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None: return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
        
        marker_array = MarkerArray()
        
        for i in range(len(ids)):
            # Depth 보정 (RGB기반 Z값 덮어쓰기)
            cx, cy = int(np.mean(corners[i][0][:,0])), int(np.mean(corners[i][0][:,1]))
            if self.latest_depth is not None and 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                d_val = self.latest_depth[cy, cx]
                if d_val > 0.0: tvecs[i][0][2] = d_val

            # TF 송출
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "camera_link" # 카메라 기준 좌표계 (TF 연결 중요!)
            t.child_frame_id = f"aruco_marker_{ids[i][0]}"
            t.transform.translation.x = float(tvecs[i][0][0])
            t.transform.translation.y = float(tvecs[i][0][1])
            t.transform.translation.z = float(tvecs[i][0][2])
            
            rmat = cv2.Rodrigues(rvecs[i])[0]
            q = rotation_matrix_to_quaternion(rmat)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q[0], q[1], q[2], q[3]
            
            self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(ArucoRealSense())
    rclpy.shutdown()

if __name__ == '__main__':
    main()