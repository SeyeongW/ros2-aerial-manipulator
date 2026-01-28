#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "control_msgs/action/gripper_command.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "aruco_markers_msgs/msg/marker_array.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using namespace std::chrono_literals;
using GripperCommand = control_msgs::action::GripperCommand;

static inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

enum class RobotState {
  WAIT_FOR_JOINT_STATE,
  WAIT_FOR_MARKER,
  APPROACH,
  DESCEND,
  GRIP_CLOSE,
  LIFT,
  MOVE_TO_PLACE,
  RELEASE_OPEN,
  RETREAT,
  DONE
};

class ArucoPickPlaceIKNode : public rclcpp::Node {
public:
  ArucoPickPlaceIKNode() : Node("aruco_pick_place_ik_node") {
    // -----------------------------
    // Topics / Frames
    // -----------------------------
    arm_topic_ = declare_parameter<std::string>("arm_topic", "/arm_controller/joint_trajectory");
    gripper_action_topic_ =
      declare_parameter<std::string>("gripper_action_topic", "/gripper_controller/gripper_cmd");

    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

    marker_topic_ = declare_parameter<std::string>("marker_topic", "/aruco/markers");
    target_marker_id_ = declare_parameter<int>("target_marker_id", 0);

    // -----------------------------
    // Robot geometry (tuned values)
    // -----------------------------
    base_off_z_ = declare_parameter<double>("base_off_z", 0.077);
    base_off_x_ = declare_parameter<double>("base_off_x", 0.012);
    L1_         = declare_parameter<double>("L1", 0.128);
    L2_         = declare_parameter<double>("L2", 0.124);
    Lw_         = declare_parameter<double>("Lw", 0.090); // NOTE: set to your real EE offset

    // -----------------------------
    // IK / Motion preferences
    // -----------------------------
    force_extend_ = declare_parameter<bool>("force_extend", false);
    extend_eps_   = declare_parameter<double>("extend_eps", 0.005);
    prefer_extended_solution_ = declare_parameter<bool>("prefer_extended_solution", true);

    min_ee_z_  = declare_parameter<double>("min_ee_z", 0.18);
    min_pitch_ = declare_parameter<double>("min_pitch", -0.2);
    max_pitch_ = declare_parameter<double>("max_pitch", 0.8);

    // -----------------------------
    // Pick parameters
    // -----------------------------
    approach_dz_ = declare_parameter<double>("approach_dz", 0.08); // approach height above marker
    grasp_dz_    = declare_parameter<double>("grasp_dz", 0.015);   // final grasp height offset
    lift_dz_     = declare_parameter<double>("lift_dz", 0.10);     // lift after grasp

    // In your IK, pitch is the end-effector pitch in the r-z plane.
    // For a simple test, keep a fixed pitch.
    pick_pitch_  = declare_parameter<double>("pick_pitch", 0.25);  // ~14 deg up
    place_pitch_ = declare_parameter<double>("place_pitch", 0.25);

    traj_duration_ = declare_parameter<double>("traj_duration", 2.0);
    settle_time_   = declare_parameter<double>("settle_time", 0.6);

    // -----------------------------
    // Place target (in base_frame)
    // -----------------------------
    place_x_ = declare_parameter<double>("place_x", 0.20);
    place_y_ = declare_parameter<double>("place_y", -0.12);
    place_z_ = declare_parameter<double>("place_z", 0.20);

    // -----------------------------
    // Gripper command values (tune for your gripper)
    // -----------------------------
    gripper_open_pos_  = declare_parameter<double>("gripper_open_pos", 0.015);
    gripper_close_pos_ = declare_parameter<double>("gripper_close_pos", -0.010);
    gripper_effort_    = declare_parameter<double>("gripper_effort", 1.0);

    // -----------------------------
    // ROS interfaces
    // -----------------------------
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    arm_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(arm_topic_, qos);
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(this, gripper_action_topic_);

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", qos,
      std::bind(&ArucoPickPlaceIKNode::cbJointStates, this, std::placeholders::_1)
    );

    marker_sub_ = create_subscription<aruco_markers_msgs::msg::MarkerArray>(
      marker_topic_, qos,
      std::bind(&ArucoPickPlaceIKNode::cbMarkers, this, std::placeholders::_1)
    );

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // State init
    current_state_ = RobotState::WAIT_FOR_JOINT_STATE;
    joint_state_received_ = false;
    command_sent_ = false;
    state_start_time_ = this->now();

    current_joints_ = {0.0, 0.0, 0.0, 0.0};

    // Timer
    timer_ = create_wall_timer(50ms, std::bind(&ArucoPickPlaceIKNode::controlLoop, this));

    RCLCPP_INFO(get_logger(), "Aruco Pick&Place IK node started.");
    RCLCPP_INFO(get_logger(), "base_frame='%s', marker_topic='%s', target_marker_id=%d",
                base_frame_.c_str(), marker_topic_.c_str(), target_marker_id_);
  }

private:
  // -----------------------------
  // Callbacks
  // -----------------------------
  void cbJointStates(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::map<std::string, double> joint_map;
    for (size_t i = 0; i < msg->name.size(); i++) {
      joint_map[msg->name[i]] = msg->position[i];
    }

    if (joint_map.count("joint1") && joint_map.count("joint2") &&
        joint_map.count("joint3") && joint_map.count("joint4")) {
      current_joints_[0] = joint_map["joint1"];
      current_joints_[1] = joint_map["joint2"];
      current_joints_[2] = joint_map["joint3"];
      current_joints_[3] = joint_map["joint4"];
      joint_state_received_ = true;
    }
  }

  void cbMarkers(const aruco_markers_msgs::msg::MarkerArray::SharedPtr msg) {
    // Find target marker by ID
    for (const auto &m : msg->markers) {
      if (m.id == target_marker_id_) {
        latest_marker_pose_ = geometry_msgs::msg::PoseStamped();
        latest_marker_pose_->header = m.header;
        latest_marker_pose_->pose = m.pose.pose;
        marker_received_ = true;
        marker_last_update_ = this->now();
        return;
      }
    }
  }

  // -----------------------------
  // Main control loop
  // -----------------------------
  void controlLoop() {
    const auto now = this->now();
    const double t = (now - state_start_time_).seconds();

    switch (current_state_) {
      case RobotState::WAIT_FOR_JOINT_STATE:
        if (joint_state_received_) {
          RCLCPP_INFO(get_logger(), "Joint states OK. Waiting for marker...");
          changeState(RobotState::WAIT_FOR_MARKER);
        }
        break;

      case RobotState::WAIT_FOR_MARKER:
        if (!marker_received_) break;

        // Require marker to be "fresh" (avoid using stale pose)
        if ((now - marker_last_update_).seconds() > 0.5) break;

        // Compute marker pose in base frame (TF)
        marker_in_base_ = transformPoseToBase(*latest_marker_pose_);
        if (!marker_in_base_) {
          // TF not ready
          break;
        }

        RCLCPP_INFO(get_logger(),
                    "Marker %d in base: x=%.3f y=%.3f z=%.3f (frame=%s)",
                    target_marker_id_,
                    marker_in_base_->pose.position.x,
                    marker_in_base_->pose.position.y,
                    marker_in_base_->pose.position.z,
                    base_frame_.c_str());

        changeState(RobotState::APPROACH);
        break;

      case RobotState::APPROACH:
        if (!command_sent_) {
          operateGripper(true); // open first
          sendIKToPose(*marker_in_base_, approach_dz_, pick_pitch_, traj_duration_);
          command_sent_ = true;
        }
        if (t > traj_duration_ + settle_time_) changeState(RobotState::DESCEND);
        break;

      case RobotState::DESCEND:
        if (!command_sent_) {
          // Refresh marker before descending (optional but safer)
          auto refreshed = marker_in_base_;
          auto latest_ok = (marker_received_ && (now - marker_last_update_).seconds() <= 0.5);
          if (latest_ok && latest_marker_pose_) {
            auto tmp = transformPoseToBase(*latest_marker_pose_);
            if (tmp) refreshed = tmp;
          }

          sendIKToPose(*refreshed, grasp_dz_, pick_pitch_, traj_duration_);
          command_sent_ = true;
        }
        if (t > traj_duration_ + settle_time_) changeState(RobotState::GRIP_CLOSE);
        break;

      case RobotState::GRIP_CLOSE:
        if (!command_sent_) {
          operateGripper(false);
          command_sent_ = true;
        }
        if (t > 1.0) changeState(RobotState::LIFT);
        break;

      case RobotState::LIFT:
        if (!command_sent_) {
          // Lift straight up in Z (same XY)
          geometry_msgs::msg::PoseStamped p = *marker_in_base_;
          p.pose.position.z += lift_dz_;
          sendIKToXYZ(p.pose.position.x, p.pose.position.y, p.pose.position.z,
                      pick_pitch_, traj_duration_);
          command_sent_ = true;
        }
        if (t > traj_duration_ + settle_time_) changeState(RobotState::MOVE_TO_PLACE);
        break;

      case RobotState::MOVE_TO_PLACE:
        if (!command_sent_) {
          // Go to place position (in base)
          sendIKToXYZ(place_x_, place_y_, place_z_, place_pitch_, traj_duration_);
          command_sent_ = true;
        }
        if (t > traj_duration_ + settle_time_) changeState(RobotState::RELEASE_OPEN);
        break;

      case RobotState::RELEASE_OPEN:
        if (!command_sent_) {
          operateGripper(true);
          command_sent_ = true;
        }
        if (t > 1.0) changeState(RobotState::RETREAT);
        break;

      case RobotState::RETREAT:
        if (!command_sent_) {
          // Retreat upward a bit
          sendIKToXYZ(place_x_, place_y_, place_z_ + 0.10, place_pitch_, traj_duration_);
          command_sent_ = true;
        }
        if (t > traj_duration_ + settle_time_) changeState(RobotState::DONE);
        break;

      case RobotState::DONE:
        // Stay idle
        break;
    }
  }

  void changeState(RobotState ns) {
    current_state_ = ns;
    state_start_time_ = this->now();
    command_sent_ = false;
  }

  // -----------------------------
  // TF helper
  // -----------------------------
  std::optional<geometry_msgs::msg::PoseStamped>
  transformPoseToBase(const geometry_msgs::msg::PoseStamped &in) {
    try {
      // Use latest available transform
      auto out = tf_buffer_->transform(in, base_frame_, tf2::durationFromSec(0.1));
      out.header.frame_id = base_frame_;
      return out;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF transform failed: %s", ex.what());
      return std::nullopt;
    }
  }

  // -----------------------------
  // High-level motion helpers
  // -----------------------------
  void sendIKToPose(const geometry_msgs::msg::PoseStamped &marker_base,
                   double dz, double pitch, double duration) {
    const double x = marker_base.pose.position.x;
    const double y = marker_base.pose.position.y;
    const double z = marker_base.pose.position.z + dz;
    sendIKToXYZ(x, y, z, pitch, duration);
  }

  void sendIKToXYZ(double x, double y, double z, double pitch, double duration) {
    // Convert (x,y) to planar dist + yaw (joint1)
    const double dist = std::sqrt(x * x + y * y);
    const double yaw  = std::atan2(y, x);

    // Safety clamps to avoid floor-collapse
    z = std::max(z, min_ee_z_);
    pitch = clamp(pitch, min_pitch_, max_pitch_);

    const bool ok = solveIKCore(dist, z, yaw, pitch, duration);
    if (!ok) {
      RCLCPP_WARN(get_logger(), "IK failed for x=%.3f y=%.3f z=%.3f dist=%.3f yaw=%.3f pitch=%.3f",
                  x, y, z, dist, yaw, pitch);
    }
  }

  // -----------------------------
  // Gripper
  // -----------------------------
  void operateGripper(bool open) {
    if (!gripper_client_->wait_for_action_server(std::chrono::seconds(1))) {
      RCLCPP_ERROR(get_logger(), "Gripper action server not available");
      return;
    }
    auto goal = GripperCommand::Goal();
    goal.command.position = open ? gripper_open_pos_ : gripper_close_pos_;
    goal.command.max_effort = gripper_effort_;

    auto opts = rclcpp_action::Client<GripperCommand>::SendGoalOptions();
    gripper_client_->async_send_goal(goal, opts);
  }

  // -----------------------------
  // IK core (same style as yours)
  // -----------------------------
  bool solveIKCore(double dist_xy, double z, double target_j1, double pitch, double duration) {
    // EE target -> planar wrist target
    double r  = dist_xy - base_off_x_;
    double zz = z - base_off_z_;

    double xw = r  - Lw_ * std::cos(pitch);
    double zw = zz - Lw_ * std::sin(pitch);

    // Optionally force near-full extension by projecting wrist to max reach circle
    if (force_extend_) {
      const double max_reach = (L1_ + L2_) - extend_eps_;
      const double d = std::sqrt(xw * xw + zw * zw);
      if (d > 1e-6) {
        xw *= (max_reach / d);
        zw *= (max_reach / d);
      } else {
        xw = max_reach;
        zw = 0.0;
      }
    }

    // 2-link planar IK
    double d2 = xw * xw + zw * zw;
    double c3 = (d2 - L1_ * L1_ - L2_ * L2_) / (2.0 * L1_ * L2_);
    c3 = clamp(c3, -1.0, 1.0);

    const double s3_pos =  std::sqrt(std::max(0.0, 1.0 - c3 * c3));
    const double s3_neg = -s3_pos;

    auto solve_with_s3 = [&](double s3) -> std::pair<double, double> {
      const double j3 = std::atan2(s3, c3);
      const double k1 = L1_ + L2_ * c3;
      const double k2 = L2_ * s3;
      const double j2 = std::atan2(zw, xw) - std::atan2(k2, k1);
      return {j2, j3};
    };

    auto [j2_1, j3_1] = solve_with_s3(s3_pos);
    auto [j2_2, j3_2] = solve_with_s3(s3_neg);

    double t_j2, t_j3;
    if (prefer_extended_solution_) {
      // Pick solution closer to straight elbow (|j3| small)
      if (std::abs(j3_1) <= std::abs(j3_2)) {
        t_j2 = j2_1; t_j3 = j3_1;
      } else {
        t_j2 = j2_2; t_j3 = j3_2;
      }
    } else {
      // Fallback heuristic
      if (j2_1 > -1.5 && j2_1 < 1.0) {
        t_j2 = j2_1; t_j3 = j3_1;
      } else {
        t_j2 = j2_2; t_j3 = j3_2;
      }
    }

    const double t_j4 = pitch - (t_j2 + t_j3);

    std::vector<double> target_q = {target_j1, t_j2, t_j3, t_j4};
    publishTrajectory(target_q, duration);
    return true;
  }

  void publishTrajectory(const std::vector<double> &target_q, double duration) {
    trajectory_msgs::msg::JointTrajectory jt;
    jt.header.stamp = rclcpp::Time(0);
    jt.joint_names = {"joint1", "joint2", "joint3", "joint4"};

    trajectory_msgs::msg::JointTrajectoryPoint p1, p2;
    p1.positions = current_joints_;
    p1.time_from_start = rclcpp::Duration::from_seconds(0.0);
    jt.points.push_back(p1);

    p2.positions = target_q;
    p2.time_from_start = rclcpp::Duration::from_seconds(duration);
    jt.points.push_back(p2);

    arm_pub_->publish(jt);
  }

private:
  // Params
  std::string arm_topic_, gripper_action_topic_;
  std::string marker_topic_;
  std::string base_frame_;
  int target_marker_id_;

  double base_off_z_, base_off_x_, L1_, L2_, Lw_;

  bool force_extend_;
  double extend_eps_;
  bool prefer_extended_solution_;

  double min_ee_z_, min_pitch_, max_pitch_;

  double approach_dz_, grasp_dz_, lift_dz_;
  double pick_pitch_, place_pitch_;
  double traj_duration_, settle_time_;

  double place_x_, place_y_, place_z_;

  double gripper_open_pos_, gripper_close_pos_, gripper_effort_;

  // State
  RobotState current_state_;
  rclcpp::Time state_start_time_;
  bool command_sent_;
  bool joint_state_received_;

  std::vector<double> current_joints_;

  // Marker data
  bool marker_received_{false};
  rclcpp::Time marker_last_update_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_marker_pose_;
  std::optional<geometry_msgs::msg::PoseStamped> marker_in_base_;

  // ROS
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<aruco_markers_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArucoPickPlaceIKNode>());
  rclcpp::shutdown();
  return 0;
}
