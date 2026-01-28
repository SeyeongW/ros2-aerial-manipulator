#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "control_msgs/action/gripper_command.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;
using GripperCommand = control_msgs::action::GripperCommand;

static inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

enum class RobotState {
  WAIT_FOR_JOINT_STATE,

  MOVE_HOME,

  // PICK sequence (fixed, no ArUco)
  MOVE_ABOVE_PICK,
  GRIPPER_OPEN_BEFORE_PICK,
  MOVE_DOWN_TO_PICK,
  GRIPPER_CLOSE_TO_GRASP,
  LIFT_AFTER_PICK,

  // PLACE sequence
  MOVE_ABOVE_PLACE,
  MOVE_DOWN_TO_PLACE,
  GRIPPER_OPEN_TO_RELEASE,
  LIFT_AFTER_PLACE,

  RETURN_HOME
};

class FixedPickPlaceIKNode : public rclcpp::Node {
public:
  FixedPickPlaceIKNode() : Node("fixed_pick_place_ik_node") {
    // ---- Topics (hard-coded, no terminal params) ----
    arm_topic_ = "/arm_controller/joint_trajectory";
    gripper_action_topic_ = "/gripper_controller/gripper_cmd";
    joint_state_topic_ = "/joint_states";

    // ---- Kinematic constants (OpenManipulator-X geometry) ----
    // Offsets & link lengths should match your URDF design intent.
    // base_off_z_ is the base height from ground to shoulder plane reference (approx 0.077 m on OM-X drawings).
    // Lw_ should match link5 -> end_effector_link fixed offset (often around 0.126 m in OM-X URDF graph).
    base_off_z_ = 0.077;
    base_off_x_ = 0.012;
    L1_ = 0.128;
    L2_ = 0.124;
    Lw_ = 0.126;

    // ---- Fixed pick/place targets (base frame assumed == link1 in your TF tree) ----
    pick_x_  = 0.200;
    pick_y_  = 0.000;

    // Place somewhere safe and higher
    place_x_ = 0.250;
    place_y_ = -0.120;

    // ---- Heights (IMPORTANT: keep fingers off the ground) ----
    // approach_z: high enough to clear everything
    // pick_z: still not "ground", keep margin; if object sits on ground, you STILL must keep EE higher
    // because EE is not finger tip. Start conservative.
    approach_z_ = 0.260; // safe approach height
    pick_z_     = 0.170; // conservative grasp height (raise if still colliding)
    place_z_    = 0.200; // place height (raise if collisions)

    // ---- Orientation ----
    // Keep pitch horizontal (0), and yaw will be computed from (x,y)
    fixed_pitch_ = 0.0;

    // ---- Joint1 yaw clamp (optional safety) ----
    // If joint1 seems not reaching, it might be a joint limit in URDF/controller.
    // Clamp wide, let controller enforce actual limits if any.
    yaw_min_ = -M_PI;
    yaw_max_ =  M_PI;

    // ---- Motion timing ----
    move_duration_ = 2.0;
    grip_wait_ = 1.0;

    // ---- Gripper positions ----
    // NOTE: Depending on your gripper controller, open/close sign can vary.
    // These are typical for OM-X simulation setups.
    gripper_open_pos_  = 0.015;
    gripper_close_pos_ = -0.010;
    gripper_effort_    = 1.0;

    current_state_ = RobotState::WAIT_FOR_JOINT_STATE;
    joint_state_received_ = false;
    command_sent_ = false;
    current_joints_ = {0.0, 0.0, 0.0, 0.0};
    state_start_time_ = this->now();

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    arm_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(arm_topic_, qos);
    gripper_action_client_ = rclcpp_action::create_client<GripperCommand>(this, gripper_action_topic_);

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, qos,
      std::bind(&FixedPickPlaceIKNode::cbJointStates, this, std::placeholders::_1)
    );

    timer_ = create_wall_timer(50ms, std::bind(&FixedPickPlaceIKNode::controlLoop, this));

    RCLCPP_INFO(get_logger(), "=== FIXED IK PICK & PLACE (NO ARUCO) ===");
    RCLCPP_INFO(get_logger(), "Pick  (%.3f, %.3f, %.3f)", pick_x_, pick_y_, pick_z_);
    RCLCPP_INFO(get_logger(), "Place (%.3f, %.3f, %.3f)", place_x_, place_y_, place_z_);
    RCLCPP_INFO(get_logger(), "Approach Z = %.3f (ground-safe)", approach_z_);
  }

private:
  void controlLoop() {
    const auto now = this->now();
    const double t = (now - state_start_time_).seconds();

    switch (current_state_) {
      case RobotState::WAIT_FOR_JOINT_STATE:
        if (joint_state_received_) {
          RCLCPP_INFO(get_logger(), ">> Joint states received. Starting routine...");
          changeState(RobotState::MOVE_HOME);
        }
        break;

      case RobotState::MOVE_HOME:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> HOME");
          // Put arm in a safe, elevated neutral pose
          moveToXYZYawPitch(0.16, 0.00, 0.24, 0.0, 0.0);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::MOVE_ABOVE_PICK);
        break;

      // -------- PICK --------
      case RobotState::MOVE_ABOVE_PICK:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> ABOVE PICK");
          moveToXYZYawPitch(pick_x_, pick_y_, approach_z_, computeYaw(pick_x_, pick_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::GRIPPER_OPEN_BEFORE_PICK);
        break;

      case RobotState::GRIPPER_OPEN_BEFORE_PICK:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> GRIPPER OPEN (before pick)");
          operateGripper(true);
          command_sent_ = true;
        }
        if (t > grip_wait_) changeState(RobotState::MOVE_DOWN_TO_PICK);
        break;

      case RobotState::MOVE_DOWN_TO_PICK:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> DOWN TO PICK (z=%.3f)", pick_z_);
          moveToXYZYawPitch(pick_x_, pick_y_, pick_z_, computeYaw(pick_x_, pick_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::GRIPPER_CLOSE_TO_GRASP);
        break;

      case RobotState::GRIPPER_CLOSE_TO_GRASP:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> GRIPPER CLOSE (grasp)");
          operateGripper(false);
          command_sent_ = true;
        }
        if (t > grip_wait_) changeState(RobotState::LIFT_AFTER_PICK);
        break;

      case RobotState::LIFT_AFTER_PICK:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> LIFT AFTER PICK");
          moveToXYZYawPitch(pick_x_, pick_y_, approach_z_, computeYaw(pick_x_, pick_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::MOVE_ABOVE_PLACE);
        break;

      // -------- PLACE --------
      case RobotState::MOVE_ABOVE_PLACE:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> ABOVE PLACE");
          moveToXYZYawPitch(place_x_, place_y_, approach_z_, computeYaw(place_x_, place_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::MOVE_DOWN_TO_PLACE);
        break;

      case RobotState::MOVE_DOWN_TO_PLACE:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> DOWN TO PLACE (z=%.3f)", place_z_);
          moveToXYZYawPitch(place_x_, place_y_, place_z_, computeYaw(place_x_, place_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::GRIPPER_OPEN_TO_RELEASE);
        break;

      case RobotState::GRIPPER_OPEN_TO_RELEASE:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> GRIPPER OPEN (release)");
          operateGripper(true);
          command_sent_ = true;
        }
        if (t > grip_wait_) changeState(RobotState::LIFT_AFTER_PLACE);
        break;

      case RobotState::LIFT_AFTER_PLACE:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> LIFT AFTER PLACE");
          moveToXYZYawPitch(place_x_, place_y_, approach_z_, computeYaw(place_x_, place_y_), fixed_pitch_);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::RETURN_HOME);
        break;

      case RobotState::RETURN_HOME:
        if (!command_sent_) {
          RCLCPP_INFO(get_logger(), ">> RETURN HOME");
          moveToXYZYawPitch(0.16, 0.00, 0.24, 0.0, 0.0);
          command_sent_ = true;
        }
        if (t > move_duration_ + 0.5) changeState(RobotState::MOVE_ABOVE_PICK);
        break;
    }
  }

  void changeState(RobotState s) {
    current_state_ = s;
    state_start_time_ = this->now();
    command_sent_ = false;
  }

  void cbJointStates(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::map<std::string, double> joint_map;
    for (size_t i = 0; i < msg->name.size(); i++) {
      if (i < msg->position.size()) joint_map[msg->name[i]] = msg->position[i];
    }

    // names depend on your robot description; this matches your previous code
    if (joint_map.count("joint1") && joint_map.count("joint2") &&
        joint_map.count("joint3") && joint_map.count("joint4")) {
      current_joints_[0] = joint_map["joint1"];
      current_joints_[1] = joint_map["joint2"];
      current_joints_[2] = joint_map["joint3"];
      current_joints_[3] = joint_map["joint4"];
      joint_state_received_ = true;
    }
  }

  double computeYaw(double x, double y) const {
    double yaw = std::atan2(y, x);
    return clamp(yaw, yaw_min_, yaw_max_);
  }

  void moveToXYZYawPitch(double x, double y, double z, double yaw, double pitch) {
    // Convert (x,y) to planar distance r for the IK core (your IK is planar in x-z with joint1 yaw)
    double dist_xy = std::sqrt(x*x + y*y);
    solveIKCore(dist_xy, z, yaw, pitch, move_duration_);
  }

  // Planar 2-link IK with wrist offset (Lw_) and base offsets.
  // Targets: joint1 yaw = target_j1, end_effector_link origin at (dist_xy, z) in base frame.
  bool solveIKCore(double r_input, double z, double target_j1, double pitch, double duration) {
    // Base offsets (model-specific)
    double r = r_input - base_off_x_;
    double zz = z - base_off_z_;

    // Wrist point (subtract Lw along end-effector direction)
    double xw = r  - Lw_ * std::cos(pitch);
    double zw = zz - Lw_ * std::sin(pitch);

    double d2 = xw*xw + zw*zw;
    double c3 = (d2 - L1_*L1_ - L2_*L2_) / (2.0 * L1_ * L2_);

    if (c3 < -1.0 || c3 > 1.0) {
      RCLCPP_WARN(get_logger(), "IK unreachable: r=%.3f z=%.3f (after offsets). Increase Z or reduce reach.", r_input, z);
      return false;
    }

    // Two solutions
    double s3a = std::sqrt(1.0 - c3*c3);
    double s3b = -s3a;

    double j3a = std::atan2(s3a, c3);
    double j3b = std::atan2(s3b, c3);

    auto solve_j2 = [&](double s3, double j3) {
      double k1 = L1_ + L2_ * c3;
      double k2 = L2_ * s3;
      return std::atan2(zw, xw) - std::atan2(k2, k1);
    };

    double j2a = solve_j2(s3a, j3a);
    double j2b = solve_j2(s3b, j3b);

    // Choose a "reasonable" elbow configuration (avoid folding under the base)
    // You can tune these if your arm prefers elbow-up/down.
    double t_j2, t_j3;
    if (j2a > -1.7 && j2a < 1.2) { t_j2 = j2a; t_j3 = j3a; }
    else                         { t_j2 = j2b; t_j3 = j3b; }

    double t_j4 = pitch - (t_j2 + t_j3);

    std::vector<double> q = {target_j1, t_j2, t_j3, t_j4};
    publishTrajectory(q, duration);
    return true;
  }

  void publishTrajectory(const std::vector<double> &target_q, double duration) {
    trajectory_msgs::msg::JointTrajectory jt;
    jt.header.stamp = rclcpp::Time(0);
    jt.joint_names = {"joint1", "joint2", "joint3", "joint4"};

    trajectory_msgs::msg::JointTrajectoryPoint p0, p1;
    p0.positions = current_joints_;
    p0.time_from_start = rclcpp::Duration::from_seconds(0.0);
    jt.points.push_back(p0);

    p1.positions = target_q;
    p1.time_from_start = rclcpp::Duration::from_seconds(duration);
    jt.points.push_back(p1);

    arm_pub_->publish(jt);
  }

  void operateGripper(bool open) {
    if (!gripper_action_client_->wait_for_action_server(std::chrono::seconds(1))) {
      RCLCPP_ERROR(get_logger(), "Gripper action server not available");
      return;
    }

    auto goal = GripperCommand::Goal();
    goal.command.position = open ? gripper_open_pos_ : gripper_close_pos_;
    goal.command.max_effort = gripper_effort_;

    auto options = rclcpp_action::Client<GripperCommand>::SendGoalOptions();
    gripper_action_client_->async_send_goal(goal, options);
  }

private:
  // Topics
  std::string arm_topic_;
  std::string gripper_action_topic_;
  std::string joint_state_topic_;

  // Geometry (meters)
  double base_off_z_{0.077};
  double base_off_x_{0.012};
  double L1_{0.128};
  double L2_{0.124};
  double Lw_{0.126};

  // Targets
  double pick_x_{0.200}, pick_y_{0.000};
  double place_x_{0.250}, place_y_{-0.120};

  // Heights
  double approach_z_{0.260};
  double pick_z_{0.170};
  double place_z_{0.200};

  // Orientation
  double fixed_pitch_{0.0};
  double yaw_min_{-M_PI}, yaw_max_{M_PI};

  // Timing
  double move_duration_{2.0};
  double grip_wait_{1.0};

  // Gripper
  double gripper_open_pos_{0.015};
  double gripper_close_pos_{-0.010};
  double gripper_effort_{1.0};

  // State
  RobotState current_state_{RobotState::WAIT_FOR_JOINT_STATE};
  rclcpp::Time state_start_time_;
  bool command_sent_{false};
  bool joint_state_received_{false};
  std::vector<double> current_joints_{0.0, 0.0, 0.0, 0.0};

  // ROS interfaces
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_action_client_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FixedPickPlaceIKNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
