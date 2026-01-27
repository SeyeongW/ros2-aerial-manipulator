#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from aruco_markers_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np


def rotation_matrix_to_quaternion(R: np.ndarray):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
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

        # Reduce potential threading/ocl issues inside OpenCV.
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(1)

        # Parameters
        self.declare_parameter("dictionary", "DICT_5X5_100")
        self.declare_parameter("marker_size", 0.06)
        self.declare_parameter("image_topic", "/camera/depth_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth_camera/camera_info")
        self.declare_parameter("annotated_image_topic", "/aruco/markers/image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("enable_pose", False)

        dict_str = str(self.get_parameter("dictionary").value)
        self.marker_size = float(self.get_parameter("marker_size").value)
        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        debug_topic = str(self.get_parameter("annotated_image_topic").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.enable_pose = bool(self.get_parameter("enable_pose").value)

        # QoS: your publisher is RELIABLE, so subscribe as RELIABLE to ensure matching.
        self.qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ArUco availability
        self.aruco_available = hasattr(cv2, "aruco")
        if not self.aruco_available:
            self.get_logger().error(
                "OpenCV ArUco module is unavailable. Install opencv-contrib-python."
            )
            self.aruco_dict = None
            self.aruco_params = None
        else:
            self.aruco_dict = self.get_aruco_dictionary(dict_str)

            # IMPORTANT:
            # Avoid cv2.aruco.DetectorParameters() and ArucoDetector in opencv-contrib-python 4.6.0.66,
            # because DetectorParameters() is known to segfault in that wheel.
            # Use the legacy factory method instead.
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.warned_encodings = set()

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, "/aruco/markers", 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 10)

        # Subscriptions (RELIABLE QoS)
        self.create_subscription(CameraInfo, info_topic, self.info_callback, self.qos_reliable)
        self.create_subscription(Image, image_topic, self.image_callback, self.qos_reliable)

        self.get_logger().info(f"Aruco node running. image_topic={image_topic}")
        if not self.enable_pose:
            self.get_logger().warn("Pose estimation is disabled (enable_pose:=true to turn on).")

    def get_aruco_dictionary(self, dict_name: str):
        aruco_dict = {
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        return cv2.aruco.getPredefinedDictionary(
            aruco_dict.get(dict_name, cv2.aruco.DICT_5X5_100)
        )

    def info_callback(self, msg: CameraInfo):
        # Only initialize once.
        if self.camera_matrix is not None:
            return

        camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        if (not np.isfinite(camera_matrix).all()) or np.allclose(camera_matrix, 0.0):
            self.get_logger().warn("Camera matrix is invalid; waiting for a valid CameraInfo.")
            return

        if len(msg.d) == 0:
            dist_coeffs = np.zeros((1, 5), dtype=np.float64)
        else:
            dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((1, -1))

        self.camera_matrix = np.ascontiguousarray(camera_matrix)
        self.dist_coeffs = np.ascontiguousarray(dist_coeffs)
        self.get_logger().info("CameraInfo received. Ready to process images.")

    def _warn_encoding_once(self, encoding: str, message: str):
        if encoding in self.warned_encodings:
            return
        self.warned_encodings.add(encoding)
        self.get_logger().warn(message)

    def _extract_buffer(self, msg: Image, channels: int):
        # Basic guards
        if msg.width == 0 or msg.height == 0:
            self._warn_encoding_once("zero_size", "Received image with zero width/height; skipping.")
            return None
        if msg.step == 0:
            self._warn_encoding_once("step_zero", "Received image with step=0; skipping.")
            return None

        expected_step = msg.width * channels
        if msg.step < expected_step:
            self._warn_encoding_once(
                "step_short",
                f"Image step({msg.step}) < width*channels({expected_step}); skipping.",
            )
            return None

        expected_bytes = msg.height * msg.step
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if buf.size < expected_bytes:
            self._warn_encoding_once(
                "buffer_short",
                f"Image data({buf.size}) < expected({expected_bytes}); skipping.",
            )
            return None

        buf = buf[:expected_bytes]
        # Reshape into (H, step_bytes), then crop to meaningful width*channels
        return buf.reshape((msg.height, msg.step))[:, :expected_step]

    def _to_gray(self, msg: Image):
        encoding = msg.encoding.lower()

        if encoding in ("mono8", "8uc1"):
            plane = self._extract_buffer(msg, 1)
            if plane is None:
                return None
            gray = plane.reshape((msg.height, msg.width))
            return np.ascontiguousarray(gray, dtype=np.uint8)

        if encoding in ("rgb8", "bgr8"):
            plane = self._extract_buffer(msg, 3)
            if plane is None:
                return None
            image = plane.reshape((msg.height, msg.width, 3))

            # Convert to RGB first for consistent conversion path
            if encoding == "bgr8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return np.ascontiguousarray(gray, dtype=np.uint8)

        if encoding in ("rgba8", "bgra8"):
            plane = self._extract_buffer(msg, 4)
            if plane is None:
                return None
            image = plane.reshape((msg.height, msg.width, 4))

            if encoding == "bgra8":
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            return np.ascontiguousarray(gray, dtype=np.uint8)

        self._warn_encoding_once(
            encoding,
            f"Unsupported encoding '{msg.encoding}'. Expected mono8/rgb8/bgr8/rgba8/bgra8.",
        )
        return None

    def image_callback(self, msg: Image):
        if not self.aruco_available or self.aruco_dict is None or self.aruco_params is None:
            return
        if self.camera_matrix is None:
            return

        gray = self._to_gray(msg)
        if gray is None:
            return

        # Legacy API path (more stable for opencv-contrib-python 4.6.0.66 than DetectorParameters()).
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or len(ids) == 0:
            return

        marker_array = MarkerArray()

        rvecs = None
        tvecs = None
        if self.enable_pose and self.dist_coeffs is not None:
            # Pose estimation requires correct camera intrinsics.
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )

        for i in range(len(ids)):
            marker_msg = Marker()
            marker_msg.header = msg.header
            marker_msg.id = int(ids[i][0])

            if tvecs is not None:
                marker_msg.pose.pose.position.x = float(tvecs[i][0][0])
                marker_msg.pose.pose.position.y = float(tvecs[i][0][1])
                marker_msg.pose.pose.position.z = float(tvecs[i][0][2])

            if rvecs is not None:
                rmat = cv2.Rodrigues(rvecs[i])[0]
                qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)
                marker_msg.pose.pose.orientation.x = qx
                marker_msg.pose.pose.orientation.y = qy
                marker_msg.pose.pose.orientation.z = qz
                marker_msg.pose.pose.orientation.w = qw

            marker_array.markers.append(marker_msg)

        self.marker_pub.publish(marker_array)

        if self.publish_debug:
            # Draw markers on an RGB debug image
            debug_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.aruco.drawDetectedMarkers(debug_rgb, corners)

            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = debug_rgb.shape[0]
            out_msg.width = debug_rgb.shape[1]
            out_msg.encoding = "rgb8"
            out_msg.step = debug_rgb.shape[1] * 3
            out_msg.data = debug_rgb.tobytes()
            self.debug_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
