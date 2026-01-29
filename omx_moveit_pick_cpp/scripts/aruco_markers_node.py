#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from aruco_markers_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

def rotation_matrix_to_quaternion(R: np.ndarray):
    """3x3 회전 행렬을 쿼터니언(x, y, z, w)으로 변환"""
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

class ArucoMarkerNode(Node):
    def __init__(self):
        super().__init__("aruco_markers_node")

        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(1)

        # [수정됨] 사용자님의 ros2 topic list에 맞춘 기본값 설정
        self.declare_parameter("dictionary", "DICT_5X5_100")
        self.declare_parameter("marker_size", 0.06)
        self.declare_parameter("image_topic", "/camera/depth_camera/image_raw")       # 수정됨
        self.declare_parameter("depth_topic", "/camera/depth_camera/depth/image_raw") # 수정됨
        self.declare_parameter("camera_info_topic", "/camera/depth_camera/camera_info") # 수정됨
        self.declare_parameter("annotated_image_topic", "/aruco/markers/image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("enable_pose", True)

        # 파라미터 로드
        dict_str = str(self.get_parameter("dictionary").value)
        self.marker_size = float(self.get_parameter("marker_size").value)
        image_topic = str(self.get_parameter("image_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        debug_topic = str(self.get_parameter("annotated_image_topic").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.enable_pose = bool(self.get_parameter("enable_pose").value)

        # QoS 설정 (Gazebo는 보통 Reliable)
        self.qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ArUco 설정
        self.aruco_available = hasattr(cv2, "aruco")
        if not self.aruco_available:
            self.get_logger().error("OpenCV ArUco module unavailable.")
            self.aruco_dict = None
            self.aruco_params = None
        else:
            self.aruco_dict = self.get_aruco_dictionary(dict_str)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_depth_image = None 

        # TF Broadcaster (C++ 노드가 마커 위치를 알 수 있게 TF 송출)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 퍼블리셔
        self.marker_pub = self.create_publisher(MarkerArray, "/aruco/markers", 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        # 서브스크립션
        self.create_subscription(CameraInfo, info_topic, self.info_callback, self.qos_reliable)
        self.create_subscription(Image, image_topic, self.image_callback, self.qos_reliable)
        self.create_subscription(Image, depth_topic, self.depth_callback, self.qos_reliable)

        self.get_logger().info(f"ArUco Node Started. Topics: {image_topic}, {depth_topic}")

    def get_aruco_dictionary(self, dict_name: str):
        aruco_dict = {
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        }
        return cv2.aruco.getPredefinedDictionary(aruco_dict.get(dict_name, cv2.aruco.DICT_5X5_100))

    def info_callback(self, msg: CameraInfo):
        if self.camera_matrix is not None:
            return
        camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((1, -1)) if len(msg.d) > 0 else np.zeros((1, 5))
        self.camera_matrix = np.ascontiguousarray(camera_matrix)
        self.dist_coeffs = np.ascontiguousarray(dist_coeffs)
        self.get_logger().info("Camera Info Received.")

    def depth_callback(self, msg: Image):
        try:
            if msg.encoding == "16UC1": # mm -> meters
                scale = 0.001
                dtype = np.uint16
            elif msg.encoding == "32FC1": # meters
                scale = 1.0
                dtype = np.float32
            else:
                return
            
            np_arr = np.frombuffer(msg.data, dtype=dtype)
            if msg.width * msg.height != np_arr.size: return
            depth_img = np_arr.reshape((msg.height, msg.width))
            self.latest_depth_image = depth_img.astype(np.float32) * scale
        except Exception:
            pass

    def get_depth_at_center(self, corners):
        if self.latest_depth_image is None: return -1.0
        c = corners[0]
        cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
        h, w = self.latest_depth_image.shape
        if 0 <= cx < w and 0 <= cy < h:
            val = self.latest_depth_image[cy, cx]
            if np.isfinite(val) and val > 0.0: return float(val)
        return -1.0

    def _to_gray(self, msg: Image):
        # 간단한 그레이스케일 변환 로직
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            # RGB8/BGR8 처리
            if "8" in msg.encoding and "c" not in msg.encoding.lower(): 
                 channels = 3
                 if buf.size == msg.width * msg.height * 3:
                     img = buf.reshape((msg.height, msg.width, 3))
                     if "bgr" in msg.encoding.lower():
                         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                     return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass
        return None

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or not self.aruco_available: return

        # OpenCV Image 변환 (cv_bridge 없이 직접 변환)
        gray = self._to_gray(msg)
        if gray is None: return

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) == 0: return

        marker_array = MarkerArray()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        for i in range(len(ids)):
            # 1. 마커 메시지 생성
            marker_msg = Marker()
            marker_msg.header = msg.header
            marker_msg.id = int(ids[i][0])

            # Depth 보정 적용
            measured_z = self.get_depth_at_center(corners[i])
            if measured_z > 0.0: tvecs[i][0][2] = measured_z

            marker_msg.pose.pose.position.x = float(tvecs[i][0][0])
            marker_msg.pose.pose.position.y = float(tvecs[i][0][1])
            marker_msg.pose.pose.position.z = float(tvecs[i][0][2])

            rmat = cv2.Rodrigues(rvecs[i])[0]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)
            marker_msg.pose.pose.orientation.x = qx
            marker_msg.pose.pose.orientation.y = qy
            marker_msg.pose.pose.orientation.z = qz
            marker_msg.pose.pose.orientation.w = qw
            
            marker_array.markers.append(marker_msg)

            # 2. [MoveIt을 위한 핵심] TF 브로드캐스팅
            # C++ 노드가 'aruco_marker_0' 좌표계를 찾을 수 있도록 TF를 쏴줍니다.
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = msg.header.frame_id # 보통 'camera_optical_frame'
            t.child_frame_id = f"aruco_marker_{int(ids[i][0])}"

            t.transform.translation.x = marker_msg.pose.pose.position.x
            t.transform.translation.y = marker_msg.pose.pose.position.y
            t.transform.translation.z = marker_msg.pose.pose.position.z
            t.transform.rotation = marker_msg.pose.pose.orientation

            self.tf_broadcaster.sendTransform(t)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()