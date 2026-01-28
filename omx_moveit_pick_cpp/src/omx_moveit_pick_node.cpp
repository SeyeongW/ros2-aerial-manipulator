#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp" 
#include "control_msgs/action/gripper_command.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "aruco_markers_msgs/msg/marker_array.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;
using GripperCommand = control_msgs::action::GripperCommand; 

enum class RobotState {
  INIT_GRIPPER,
  SCANNING,       
  EXECUTE_ALIGN,  
  EXECUTE_REACH,  
  EXECUTE_DOWN,   
  GRASPING,       
  LIFT,           
  MOVE_PLACE,     
  RELEASE         
};

static inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

class PickAndPlaceNode : public rclcpp::Node {
public:
  PickAndPlaceNode() : Node("omx_pick_place_node") {
    base_frame_   = declare_parameter<std::string>("base_frame", "link1");
    target_id_    = static_cast<unsigned int>(declare_parameter<int>("target_id", 0));
    marker_topic_ = declare_parameter<std::string>("marker_topic", "/aruco/markers");
    arm_topic_    = declare_parameter<std::string>("arm_topic", "/arm_controller/joint_trajectory");
    
    gripper_action_topic_ = declare_parameter<std::string>("gripper_action_topic", "/gripper_controller/gripper_cmd");

    // [하드웨어 스펙]
    base_off_z_ = 0.077; base_off_x_ = 0.012; 
    L1_ = 0.128; L2_ = 0.124; Lw_ = 0.126; 

    current_state_ = RobotState::INIT_GRIPPER;
    state_start_time_ = this->now();
    
    // [전략 변수]
    locked_yaw_ = 0.0;
    locked_dist_ = 0.0;
    locked_pitch_ = 0.0; 
    
    command_sent_ = false;
    first_scan_ = true;
    current_joints_ = {0.0, -1.0, 0.3, 0.7}; 

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.best_effort();
    
    // Arm은 기존대로 Publisher 사용
    arm_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(arm_topic_, qos);
    
    // [NEW] Gripper는 Action Client 사용!
    gripper_action_client_ = rclcpp_action::create_client<GripperCommand>(this, gripper_action_topic_);

    sub_ = create_subscription<aruco_markers_msgs::msg::MarkerArray>(
      marker_topic_, 10,
      std::bind(&PickAndPlaceNode::cbMarkers, this, std::placeholders::_1)
    );

    timer_ = this->create_wall_timer(
      50ms, std::bind(&PickAndPlaceNode::controlLoop, this));

    RCLCPP_INFO(get_logger(), "=== MODE: Action Client Gripper (Correct Way) ===");
  }

private:
  void controlLoop() {
    auto now = this->now();
    if (state_start_time_.get_clock_type() != now.get_clock_type()) state_start_time_ = now;
    double time_in_state = (now - state_start_time_).seconds();

    switch (current_state_) {
      case RobotState::INIT_GRIPPER:
        if (!command_sent_) {
            operateGripper(true);
            RCLCPP_INFO(get_logger(), ">> INIT: Action Open Sent");
            command_sent_ = true;
        }
        if (time_in_state > 2.0) changeState(RobotState::SCANNING);
        break;

      case RobotState::SCANNING:
        performWideScanning();
        break;

      // 1. 회전
      case RobotState::EXECUTE_ALIGN:
        if (!command_sent_) {
             double align_height = (locked_pitch_ == 0.0) ? 0.15 : 0.20;
             solveAndMoveLockYaw(locked_dist_, 0.0, saved_target_z_ + align_height, locked_yaw_, locked_pitch_, 1.5);
             RCLCPP_INFO(get_logger(), ">> [1/3] ALIGNING...");
             command_sent_ = true;
        }
        if (time_in_state > 1.6) changeState(RobotState::EXECUTE_REACH);
        break;

      // 2. 접근
      case RobotState::EXECUTE_REACH:
        if (!command_sent_) {
            double approach_height;
            if (locked_pitch_ == 0.0) approach_height = 0.02; 
            else approach_height = 0.15; 

            solveAndMoveLockYaw(locked_dist_, 0.0, saved_target_z_ + approach_height, locked_yaw_, locked_pitch_, 1.5); 
            RCLCPP_INFO(get_logger(), ">> [2/3] REACHING...");
            command_sent_ = true;
        }
        if (time_in_state > 1.6) changeState(RobotState::EXECUTE_DOWN);
        break;

      // 3. 하강
      case RobotState::EXECUTE_DOWN:
        if (!command_sent_) {
            double floor_target_z = 0.01; 
            if (locked_pitch_ == 0.0) floor_target_z = saved_target_z_; 

            solveAndMoveLockYaw(locked_dist_, 0.0, floor_target_z, locked_yaw_, locked_pitch_, 1.5); 
            RCLCPP_INFO(get_logger(), ">> [3/3] LANDING...");
            command_sent_ = true;
        }
        if (time_in_state > 2.0) changeState(RobotState::GRASPING); 
        break;

      // 4. 잡기
      case RobotState::GRASPING:
        if (!command_sent_) {
            operateGripper(false); // Close (Action)
            RCLCPP_INFO(get_logger(), ">> GRAB ACTION SENT!");
            command_sent_ = true;
        }
        if (time_in_state > 2.5) changeState(RobotState::LIFT);
        break;

      // 5. 들기
      case RobotState::LIFT:
        if (!command_sent_) {
            solveAndMoveLockYaw(locked_dist_, 0.0, saved_target_z_ + 0.25, locked_yaw_, locked_pitch_, 2.0);
            RCLCPP_INFO(get_logger(), ">> LIFT");
            command_sent_ = true;
        }
        if (time_in_state > 2.2) changeState(RobotState::MOVE_PLACE);
        break;

      // 6. 이동
      case RobotState::MOVE_PLACE:
        if (!command_sent_) {
            solveAndMove(0.00, -0.20, 0.15, -1.57, 2.5); 
            RCLCPP_INFO(get_logger(), ">> MOVING");
            command_sent_ = true;
        }
        if (time_in_state > 3.0) changeState(RobotState::RELEASE);
        break;

      // 7. 놓기
      case RobotState::RELEASE:
        if (!command_sent_) {
            operateGripper(true); // Open (Action)
            RCLCPP_INFO(get_logger(), ">> RELEASE ACTION SENT");
            command_sent_ = true;
        }
        if (time_in_state > 2.0) {
            changeState(RobotState::SCANNING); 
        }
        break;
    }
  }

  void changeState(RobotState new_state) {
    current_state_ = new_state;
    state_start_time_ = this->now();
    command_sent_ = false; 
  }

  void cbMarkers(const aruco_markers_msgs::msg::MarkerArray::SharedPtr msg) {
    if (current_state_ != RobotState::SCANNING) return;

    for (const auto &m : msg->markers) {
      if (m.id == target_id_) {
        geometry_msgs::msg::PoseStamped cam_pose, base_pose;
        cam_pose.header = m.header;
        cam_pose.pose = m.pose.pose;

        try {
          if (tf_buffer_->canTransform(base_frame_, m.header.frame_id, tf2::TimePointZero)) {
             base_pose = tf_buffer_->transform(cam_pose, base_frame_, tf2::durationFromSec(0.1));
             
             double tx = base_pose.pose.position.x;
             double ty = base_pose.pose.position.y;
             double tz = base_pose.pose.position.z;
             double dist = std::sqrt(tx*tx + ty*ty);

             if (tz < -0.05) tz = 0.0; 
             if (dist < 0.05 || dist > 0.45) continue;

             saved_target_x_ = tx;
             saved_target_y_ = ty;
             saved_target_z_ = tz;
             
             locked_dist_ = dist;
             locked_yaw_ = std::atan2(ty, tx);

             if (dist > 0.22) {
                 locked_pitch_ = 0.0; 
                 RCLCPP_INFO(get_logger(), "⚡ FOUND (%.2fm) -> HORIZONTAL", dist);
             } else {
                 locked_pitch_ = -1.57; 
                 RCLCPP_INFO(get_logger(), "⚡ FOUND (%.2fm) -> VERTICAL", dist);
             }

             changeState(RobotState::EXECUTE_ALIGN);
          }
        } catch (tf2::TransformException &e) {}
        break;
      }
    }
  }

  void performWideScanning() {
      auto current_time = this->now();
      if (first_scan_) { last_scan_time_ = current_time; first_scan_ = false; }
      if ((current_time - last_scan_time_).seconds() < 1.0) return;

      double t = current_time.seconds();
      std::vector<double> scan_pose = {
          1.5 * std::sin(t * 0.5), 
          -0.6, 
          0.3, 
          1.4 + 0.5 * std::sin(t * 0.8)
      };
      publishArm(scan_pose, 1.0);
      last_scan_time_ = current_time;
  }

  bool solveAndMove(double x, double y, double z, double pitch, double duration) {
      double j1 = std::atan2(y, x);
      return solveIKCore(std::sqrt(x*x + y*y), 0, z, j1, pitch, duration);
  }

  bool solveAndMoveLockYaw(double dist_xy, double y_dummy, double z, double fixed_yaw, double pitch, double duration) {
      return solveIKCore(dist_xy, 0, z, fixed_yaw, pitch, duration);
  }

  bool solveIKCore(double r_input, double y_dummy, double z, double target_j1, double pitch, double duration) {
      double r = r_input - base_off_x_;
      double zz = z - base_off_z_;
      
      double xw = r - Lw_ * std::cos(pitch);
      double zw = zz - Lw_ * std::sin(pitch);
      
      double d2 = xw*xw + zw*zw;
      double c3 = (d2 - L1_*L1_ - L2_*L2_) / (2.0 * L1_ * L2_);

      if (c3 < -1.0 || c3 > 1.0 || std::isnan(c3)) return false;

      double s3 = std::sqrt(std::max(0.0, 1.0 - c3*c3));
      double j3 = std::atan2(s3, c3);
      double k1 = L1_ + L2_ * c3;
      double k2 = L2_ * s3;
      double j2 = std::atan2(zw, xw) - std::atan2(k2, k1);
      double j4 = pitch - (j2 + j3); 

      j2 = clamp(j2, -2.0, 2.0);
      j3 = clamp(j3, -2.2, 2.2);
      j4 = clamp(j4, -2.5, 2.5);

      std::vector<double> target_joints = {target_j1, j2, j3, j4};
      publishTrajectory(target_joints, duration); 
      return true;
  }

  void publishTrajectory(const std::vector<double> &target_q, double duration) {
      trajectory_msgs::msg::JointTrajectory jt;
      jt.header.stamp = rclcpp::Time(0); 
      jt.joint_names = {"joint1", "joint2", "joint3", "joint4"};

      trajectory_msgs::msg::JointTrajectoryPoint p1;
      p1.positions = current_joints_;
      p1.time_from_start = rclcpp::Duration::from_seconds(0.0);
      jt.points.push_back(p1);

      trajectory_msgs::msg::JointTrajectoryPoint p2;
      p2.positions = target_q;
      p2.time_from_start = rclcpp::Duration::from_seconds(duration);
      jt.points.push_back(p2);

      arm_pub_->publish(jt);
      current_joints_ = target_q; 
  }
  
  void publishArm(const std::vector<double> &q, double duration) {
    trajectory_msgs::msg::JointTrajectory jt;
    jt.header.stamp = rclcpp::Time(0); 
    jt.joint_names = {"joint1", "joint2", "joint3", "joint4"};
    trajectory_msgs::msg::JointTrajectoryPoint p;
    p.positions = q;
    p.time_from_start = rclcpp::Duration::from_seconds(duration);
    jt.points.push_back(p);
    arm_pub_->publish(jt);
    current_joints_ = q;
  }

  void operateGripper(bool open) {
    if (!gripper_action_client_->wait_for_action_server(std::chrono::seconds(1))) {
      RCLCPP_ERROR(get_logger(), "Action server not available after waiting");
      return;
    }

    auto goal_msg = GripperCommand::Goal();
    
    if (open) {
        goal_msg.command.position = 0.015;
    } else {
        goal_msg.command.position = 0.0;
    }
    
    goal_msg.command.max_effort = 1.0;
    auto send_goal_options = rclcpp_action::Client<GripperCommand>::SendGoalOptions();
    gripper_action_client_->async_send_goal(goal_msg, send_goal_options);
  }

  std::string base_frame_, marker_topic_, arm_topic_, gripper_action_topic_;
  unsigned int target_id_;
  double base_off_z_, base_off_x_, L1_, L2_, Lw_;
  RobotState current_state_;
  rclcpp::Time state_start_time_, last_scan_time_; 
  bool first_scan_, target_found_, command_sent_; 
  double saved_target_x_, saved_target_y_, saved_target_z_;
  double locked_yaw_, locked_dist_, locked_pitch_;
  std::vector<double> current_joints_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_action_client_;
  
  rclcpp::Subscription<aruco_markers_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PickAndPlaceNode>());
  rclcpp::shutdown();
  return 0;
}