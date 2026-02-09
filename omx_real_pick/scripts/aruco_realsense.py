#!/usr/bin/env python3
# =========================================
# aruco_subscriber_node.py (FULL, MODIFIED)
#
# 목표:
# - Python은 "인식만" 담당한다.
# - 로봇 움직임(approach/grasp)은 C++ MoveIt 노드가 담당한다.
#
# Publish:
# 1) /aruco/target_visible : std_msgs/Bool
# 2) /aruco/target_pose    : geometry_msgs/PoseStamped  (카메라 프레임 기준)
# 3) /aruco/result_image   : Image (디버그)
#
# Optional:
# - TF broadcast (target marker only)
# =========================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import Bool

import cv2
import cv2.aruco as aruco
import numpy as np
import tf2_ros


def rotation_matrix_to_quaternion(R: np.ndarray):
    """
    OpenCV Rodrigues로 얻은 회전행렬(R)을 ROS quaternion(x,y,z,w)로 변환.
    (표준적인 변환 로직)
    """
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [float(qx), float(qy), float(qz), float(qw)]


class ArucoSubscriberNode(Node):
    def __init__(self):
        super().__init__("aruco_subscriber_node")

        # ---------------------------------
        # (중요) 시뮬레이션이면 use_sim_time=True
        # ---------------------------------
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # ---------------------------------
        # 파라미터 (런치/CLI에서 바꿀 수 있게)
        # ---------------------------------
        self.declare_parameter("marker_size", 0.06)  # 마커 한 변 길이 (m)
        self.declare_parameter("frame_id", "camera_link")  # 이미지/포즈 기준 프레임(카메라 프레임)
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        # "타겟" 마커 ID (움직일 대상)
        self.declare_parameter("target_id", 23)

        self.marker_size = float(self.get_parameter("marker_size").value)
        self.base_frame_id = str(self.get_parameter("frame_id").value)
        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        self.target_id = int(self.get_parameter("target_id").value)

        # ---------------------------------
        # 오프셋 (카메라-로봇 외부보정이 확실할 때만 사용)
        # ---------------------------------
        self.rgb_offset_x = 0.0
        self.rgb_offset_y = 0.0
        self.rgb_offset_z = 0.0

        # ---------------------------------
        # ArUco 다중 딕셔너리 탐지(네가 쓰던 방식 유지)
        # ---------------------------------
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

        # ---------------------------------
        # solvePnP용 마커 코너 좌표(마커 로컬 프레임, z=0 평면)
        # - 코너 순서: (좌상, 우상, 우하, 좌하) 로 가정
        # ---------------------------------
        ms_half = self.marker_size / 2.0
        self.marker_obj_points = np.array([
            [-ms_half,  ms_half, 0.0],
            [ ms_half,  ms_half, 0.0],
            [ ms_half, -ms_half, 0.0],
            [-ms_half, -ms_half, 0.0]
        ], dtype=np.float32)

        # ---------------------------------
        # QoS: 카메라 이미지에 적합한 BEST_EFFORT
        # ---------------------------------
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---------------------------------
        # CameraInfo에서 실제 K/D를 받아서 사용 (fake intrinsics 금지)
        # ---------------------------------
        self.camera_matrix = None
        self.dist_coeffs = None
        self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)

        # 이미지 구독
        self.create_subscription(Image, image_topic, self.image_callback, qos_policy)

        # ---------------------------------
        # Publisher들
        # ---------------------------------
        self.target_visible_pub = self.create_publisher(Bool, "/aruco/target_visible", 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, "/aruco/target_pose", 10)
        self.image_res_pub = self.create_publisher(Image, "/aruco/result_image", 10)

        # TF (선택)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.cv_bridge = CvBridge()

        self.get_logger().info(
            f"✅ ArUco Node Started | image={image_topic}, info={info_topic}, target_id={self.target_id}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        """
        CameraInfo는 보통 한 번만 받아도 됨.
        K(3x3)와 D(distortion)를 저장한다.
        """
        if self.camera_matrix is not None:
            return

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        # K가 유효하지 않으면 대기
        if not np.isfinite(K).all() or np.allclose(K, 0.0):
            self.get_logger().warn("CameraInfo K invalid; waiting...")
            return

        self.camera_matrix = K

        # distortion
        if len(msg.d) > 0:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.get_logger().info("✅ CameraInfo received. Using REAL intrinsics (K/D).")

    def image_callback(self, msg: Image):
        """
        매 프레임:
        1) ArUco 탐지 (여러 딕셔너리 돌림)
        2) target_id(예: 23)만 solvePnP 수행
        3) 유효하면 /aruco/target_pose, /aruco/target_visible 발행
        """
        # CameraInfo 없으면 진행 불가
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn_throttle(1.0, "Waiting for CameraInfo...")
            return

        # ROS Image -> OpenCV
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy()

        # 타겟 탐지 결과
        target_found = False
        best_pose = None  # PoseStamped
        best_dict_name = None
        best_z = None

        # ---------------------------------
        # 딕셔너리별로 탐지 시도
        # ---------------------------------
        for dict_name, detector in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None or len(ids) == 0:
                continue

            # 디버그용 마커 박스 표시
            aruco.drawDetectedMarkers(vis_frame, corners, ids)

            # ---------------------------------
            # 검출된 id들 중에서 target_id만 처리
            # ---------------------------------
            for i in range(len(ids)):
                current_id = int(ids[i][0])
                if current_id != self.target_id:
                    continue

                # solvePnP: IPPE_SQUARE는 평면 사각형 마커에 유리
                ok, rvec, tvec = cv2.solvePnP(
                    self.marker_obj_points,
                    corners[i],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                # tvec: 카메라 기준 마커 위치 (m)
                px = float(tvec[0][0]) + self.rgb_offset_x
                py = float(tvec[1][0]) + self.rgb_offset_y
                pz = float(tvec[2][0]) + self.rgb_offset_z

                # 깊이 sanity check (너무 가깝거나/0이하/너무 멀면 버림)
                if (not np.isfinite(pz)) or (pz <= 0.02) or (pz > 2.0):
                    continue

                rmat = cv2.Rodrigues(rvec)[0]
                qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)

                # PoseStamped 생성 (카메라 프레임 기준)
                ps = PoseStamped()
                ps.header = msg.header

                # header.frame_id가 비어있으면 파라미터 frame_id 사용
                if not ps.header.frame_id:
                    ps.header.frame_id = self.base_frame_id

                ps.pose.position.x = px
                ps.pose.position.y = py
                ps.pose.position.z = pz
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw

                # 여러 번 검출되면 "가장 가까운(z가 작은)" 것을 선택하는 방식(안정성)
                if (best_pose is None) or (pz < best_z):
                    best_pose = ps
                    best_dict_name = dict_name
                    best_z = pz
                    target_found = True

        # ---------------------------------
        # 디버그 이미지 발행
        # ---------------------------------
        if target_found and best_pose is not None:
            # 텍스트 표시
            info_text = f"TARGET ID:{self.target_id} ({best_dict_name}) z:{best_z:.2f}"
            cv2.putText(
                vis_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                vis_frame, f"TARGET ID:{self.target_id} not found", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8")
        vis_msg.header = msg.header
        self.image_res_pub.publish(vis_msg)

        # ---------------------------------
        # 핵심 출력 1) visible 신호
        # ---------------------------------
        vis = Bool()
        vis.data = bool(target_found)
        self.target_visible_pub.publish(vis)

        # ---------------------------------
        # 핵심 출력 2) target_pose
        # ---------------------------------
        if target_found and best_pose is not None:
            self.target_pose_pub.publish(best_pose)

            # ---------------------------------
            # (선택) TF broadcast: camera_frame -> aruco_marker_<ID>
            # - C++ 쪽에서 marker TF lookup을 원하면 유용
            # - 하지만 이번 구조에서는 topic 기반으로도 충분함
            # ---------------------------------
            t = TransformStamped()
            t.header = best_pose.header
            t.child_frame_id = f"aruco_marker_{self.target_id}"
            t.transform.translation.x = best_pose.pose.position.x
            t.transform.translation.y = best_pose.pose.position.y
            t.transform.translation.z = best_pose.pose.position.z
            t.transform.rotation = best_pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)


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


if __name__ == "__main__":
    main()
