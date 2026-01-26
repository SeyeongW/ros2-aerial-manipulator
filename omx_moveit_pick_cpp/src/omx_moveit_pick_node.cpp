#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "aruco_markers_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

class DirectServoPicker : public rclcpp::Node {
public:
  DirectServoPicker() : Node("direct_servo_picker") {
    // 파라미터
    base_frame_ = declare_parameter<std::string>("base_frame", "link1");
    target_id_ = static_cast<unsigned int>(declare_parameter<int>("target_id", 0));
    
    // 퍼블리셔
    arm_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/arm_controller/joint_trajectory", 10);
    gripper_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/gripper_controller/joint_trajectory", 10);

    // TF 리스너
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 마커 구독
    sub_ = create_subscription<aruco_markers_msgs::msg::MarkerArray>(
        "/aruco/markers", 10, std::bind(&DirectServoPicker::cbMarkers, this, std::placeholders::_1));

    // 스캔 타이머 (3.5초마다 실행 - 천천히 움직이기 위함)
    scan_timer_ = create_wall_timer(3500ms, std::bind(&DirectServoPicker::scanRoutine, this));
    last_detection_time_ = this->now();

    RCLCPP_INFO(get_logger(), "Direct Servo Picker Started (Wide Scan Mode)");

    // 초기화: 그리퍼 열고 정면 하방 응시
    sendGripperCommand(0.01);
    sendArmCommand({0.0, -0.8, 0.3, 1.5}, 3.0); 
  }

private:
  void cbMarkers(const aruco_markers_msgs::msg::MarkerArray::SharedPtr msg) {
    if (is_grasping_) return; 

    // 1. 타겟 찾기
    const aruco_markers_msgs::msg::Marker *target = nullptr;
    for (const auto &m : msg->markers) {
      if (m.id == target_id_) { target = &m; break; }
    }
    
    if (!target) return; 

    // 타겟 발견!
    last_detection_time_ = this->now();
    scan_step_ = 0; 

    // 2. 좌표 변환
    geometry_msgs::msg::PoseStamped target_pose;
    if (!getMarkerPose(*target, target_pose)) return;

    double x = target_pose.pose.position.x;
    double y = target_pose.pose.position.y;
    double dist = std::sqrt(x*x + y*y);

    // 3. 거리별 행동 (State Machine)
    if (dist > 0.35 || dist < 0.12) {
       trackTarget(x, y, dist);
       return;
    }

    // 사정권 진입 (Sweet Spot: 0.15 ~ 0.28m) & 중앙 조준 완료 시 잡기
    if (std::abs(y) < 0.03) {
        executeGraspSequence(x, y);
    } else {
        trackTarget(x, y, dist);
    }
  }

  void trackTarget(double x, double y, double dist) {
    double j1 = std::atan2(y, x);
    double j2, j3, j4;

    if (dist < 0.15) { 
        j2 = -0.1; j3 = -0.6; j4 = 1.6; 
    } else if (dist < 0.22) {
        j2 = 0.0; j3 = -0.2; j4 = 1.2;
    } else {
        j2 = 0.2; j3 = 0.1; j4 = 0.8;
    }

    // 추적은 0.5초로 약간 부드럽게
    sendArmCommand({j1, j2, j3, j4}, 0.5);
  }

  void executeGraspSequence(double x, double y) {
    is_grasping_ = true;
    RCLCPP_INFO(get_logger(), "GRASP TRIGGERED! (Pos: %.2f, %.2f)", x, y);

    double j1 = std::atan2(y, x);

    // 접근
    sendArmCommand({j1, 0.1, 0.1, 1.0}, 1.0);
    rclcpp::sleep_for(1200ms);

    // 잡기
    sendGripperCommand(-0.01); 
    rclcpp::sleep_for(1000ms);

    // 들기
    sendArmCommand({j1, -0.5, -0.3, 0.5}, 2.0);
    rclcpp::sleep_for(2500ms);

    // 복귀
    sendArmCommand({0.0, -0.8, 0.3, 1.5}, 2.0);
    rclcpp::sleep_for(2000ms);

    // 놓기 (테스트용)
    sendGripperCommand(0.01);
    rclcpp::sleep_for(1000ms);

    is_grasping_ = false;
    RCLCPP_INFO(get_logger(), "Routine Finished. Scanning...");
  }

  // [수정됨] 상하좌우 광역 스캔 루틴
  void scanRoutine() {
    if (is_grasping_) return;
    
    // 마지막 발견 후 2초 지났으면 스캔 시작
    auto duration = this->now() - last_detection_time_;
    if (duration.seconds() < 2.0) return;

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000, "Scanning Wide Area...");

    // 스캔 패턴 (J1:좌우, J2/J3/J4:상하)
    std::vector<std::vector<double>> patterns = {
        // 1. 아래쪽 훑기 (기존) - 발 밑 확인
        { 0.0, -0.8,  0.3,  1.5},  // 중앙 아래
        { 0.8, -0.8,  0.3,  1.5},  // 왼쪽 아래
        {-0.8, -0.8,  0.3,  1.5},  // 오른쪽 아래
        
        // 2. [추가] 정면(전방) 훑기 - 고개를 듬
        { 0.0,  0.0, -0.2,  1.0},  // 중앙 전방 (멀리 봄)
        { 0.8,  0.0, -0.2,  1.0},  // 왼쪽 전방
        {-0.8,  0.0, -0.2,  1.0}   // 오른쪽 전방
    };

    int idx = scan_step_ % patterns.size();
    
    // [속도 조절] 3.0초 동안 천천히 이동 (카메라 블러 방지)
    sendArmCommand(patterns[idx], 3.0); 
    scan_step_++;
  }

  bool getMarkerPose(const aruco_markers_msgs::msg::Marker &m, geometry_msgs::msg::PoseStamped &out) {
    try {
      geometry_msgs::msg::PoseStamped in;
      in.header = m.header;
      in.pose = m.pose.pose;
      auto tf = tf_buffer_->lookupTransform(base_frame_, m.header.frame_id, tf2::TimePointZero);
      tf2::doTransform(in, out, tf);
      return true;
    } catch (tf2::TransformException &ex) {
      return false;
    }
  }

  void sendArmCommand(std::vector<double> positions, double duration) {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.joint_names = {"joint1", "joint2", "joint3", "joint4"};
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = positions;
    point.time_from_start = rclcpp::Duration::from_seconds(duration);
    msg.points.push_back(point);
    arm_pub_->publish(msg);
  }

  void sendGripperCommand(double pos) {
    trajectory_msgs::msg::JointTrajectory msg;
    msg.joint_names = {"gripper"};
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {pos};
    point.time_from_start = rclcpp::Duration::from_seconds(1.0);
    msg.points.push_back(point);
    gripper_pub_->publish(msg);
  }

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_, gripper_pub_;
  rclcpp::Subscription<aruco_markers_msgs::msg::MarkerArray>::SharedPtr sub_;
  
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr scan_timer_;
  rclcpp::Time last_detection_time_;

  std::string base_frame_;
  unsigned target_id_;
  int scan_step_ = 0;
  bool is_grasping_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DirectServoPicker>());
  rclcpp::shutdown();
  return 0;
}