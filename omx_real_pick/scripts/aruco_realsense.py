#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers 
import cv2
import cv2.aruco as aruco
import numpy as np
import tf2_ros

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

class ArucoSubscriberNode(Node):
    def __init__(self):
        super().__init__("aruco_subscriber_node")
        
        # 파라미터
        self.declare_parameter("marker_size", 0.06)
        self.declare_parameter("frame_id", "camera_link") # 실제 카메라 링크 이름 확인 필요
        
        self.marker_size = self.get_parameter("marker_size").value
        self.base_frame_id = self.get_parameter("frame_id").value

        # [오프셋] RGB 카메라 -> 중심 링크 (필요시 조정)
        self.rgb_offset_x = 0.0
        self.rgb_offset_y = 0.0
        self.rgb_offset_z = 0.0

        # ArUco 사전 설정
        self.dict_collection = {
            "6x6": aruco.DICT_6X6_250,
            "5x5": aruco.DICT_5X5_100,
            "4x4": aruco.DICT_4X4_50,
            "ORIG": aruco.DICT_ARUCO_ORIGINAL
        }
        self.detectors = []
        for name, dict_enum in self.dict_collection.items():
            d = aruco.getPredefinedDictionary(dict_enum)
            p = aruco.DetectorParameters()
            det = aruco.ArucoDetector(d, p)
            self.detectors.append((name, det))

        # 3D 좌표용 코너 포인트
        ms_half = self.marker_size / 2.0
        self.marker_obj_points = np.array([
            [-ms_half,  ms_half, 0],
            [ ms_half,  ms_half, 0],
            [ ms_half, -ms_half, 0],
            [-ms_half, -ms_half, 0]
        ], dtype=np.float32)

        # QoS 설정 (카메라 데이터 유실 방지)
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # [핵심] 이제 카메라를 직접 열지 않고 구독(Subscribe)합니다.
        # 리얼센스 기본 토픽: /camera/color/image_raw (또는 /camera/image_raw)
        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw", 
            self.image_callback,
            qos_policy
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(ArucoMarkers, "/aruco/markers", 10)
        self.image_res_pub = self.create_publisher(Image, "/aruco/result_image", 10)
        self.cv_bridge = CvBridge()

        # 카메라 매트릭스 (임시값, CameraInfo 구독하면 더 좋음)
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))

        self.get_logger().info("✅ ArUco Subscriber Node Started! Listening to /camera/color/image_raw")

    def image_callback(self, msg):
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy()
        
        marker_msg = ArucoMarkers()
        marker_msg.header = msg.header # 원본 시간/프레임ID 계승
        
        # 만약 원본 헤더의 frame_id가 비어있다면 강제 할당
        if not marker_msg.header.frame_id:
            marker_msg.header.frame_id = self.base_frame_id

        found_any = False

        for name, detector in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                found_any = True
                aruco.drawDetectedMarkers(vis_frame, corners, ids)

                for i in range(len(ids)):
                    success, rvec, tvec = cv2.solvePnP(self.marker_obj_points, corners[i], 
                                                       self.camera_matrix, self.dist_coeffs, 
                                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not success: continue

                    # 좌표 변환
                    px = float(tvec[0][0]) + self.rgb_offset_x
                    py = float(tvec[1][0]) + self.rgb_offset_y
                    pz = float(tvec[2][0]) + self.rgb_offset_z

                    current_id = int(ids[i][0])
                    marker_msg.marker_ids.append(current_id)

                    pose = Pose()
                    pose.position.x = px; pose.position.y = py; pose.position.z = pz
                    
                    rmat = cv2.Rodrigues(rvec)[0]
                    q = rotation_matrix_to_quaternion(rmat)
                    pose.orientation.x = q[0]; pose.orientation.y = q[1]
                    pose.orientation.z = q[2]; pose.orientation.w = q[3]
                    marker_msg.poses.append(pose)

                    # TF 송출
                    t = TransformStamped()
                    t.header = marker_msg.header
                    t.child_frame_id = f"aruco_marker_{current_id}"
                    t.transform.translation.x = px
                    t.transform.translation.y = py
                    t.transform.translation.z = pz
                    t.transform.rotation = pose.orientation
                    self.tf_broadcaster.sendTransform(t)

                    # 화면 표시
                    c = corners[i][0]
                    text_pos = (int(c[0][0]), int(c[0][1]) - 10)
                    info_text = f"ID:{current_id}({name}) z:{pz:.2f}"
                    cv2.putText(vis_frame, info_text, text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 발행
        vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8")
        vis_msg.header = msg.header
        self.image_res_pub.publish(vis_msg)

        if found_any:
            self.marker_pub.publish(marker_msg)

def main():
    rclpy.init()
    node = ArucoSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()