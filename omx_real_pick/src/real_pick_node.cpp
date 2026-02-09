// ===============================
// omx_real_picker.cpp (FULL)
// Fixes applied:
// 1) Force use_sim_time=true (consistent with Gazebo /clock)
// 2) Use NodeOptions(automatically_declare_parameters_from_overrides=true)
// 3) TF lookup with robust timeout + stamp freshness check
// 4) Before every move(): setStartStateToCurrentState(), clearPoseTargets()
// 5) Clamp marker position to sane workspace ranges (reject crazy tvec)
// 6) Fix gripper close value (avoid negative if your gripper joint doesn't allow it)
// 7) Stop "keep going after abort" behavior (fail-safe logic)
// ===============================

#include <memory>
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/gripper_command.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

using GripperCommand = control_msgs::action::GripperCommand;
using namespace std::chrono_literals;

static inline double clampd(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

// NOTE: Many OM-X gripper configs expect joint value >= 0.0 for closing.
// If yours is different, adjust CLOSE_POS below after checking joint limits in URDF/SRDF.
static constexpr double GRIP_OPEN_POS  = 0.019;
static constexpr double GRIP_CLOSE_POS = 0.000;   // safer than negative
static constexpr double GRIP_EFFORT    = 0.5;

static void operateGripper(
  const rclcpp::Node::SharedPtr& node,
  const rclcpp_action::Client<GripperCommand>::SharedPtr& client,
  double pos
) {
  if (!client->wait_for_action_server(2s)) {
    RCLCPP_ERROR(node->get_logger(), "Gripper action server not available");
    return;
  }
  auto goal = GripperCommand::Goal();
  goal.command.position = pos;
  goal.command.max_effort = GRIP_EFFORT;
  client->async_send_goal(goal);
  rclcpp::sleep_for(1s);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // Receive overrides + enable sim time from launch/CLI, but also default true for Gazebo.
  auto node = std::make_shared<rclcpp::Node>(
    "omx_real_picker",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  // Force sim time ON (Gazebo publishes /clock). If you're on real robot, set false.
  node->declare_parameter("use_sim_time", true);
  node->set_parameter(rclcpp::Parameter("use_sim_time", true));

  auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  exec->add_node(node);
  std::thread spinner([&exec](){ exec->spin(); });

  // TF
  auto tf_buffer   = std::make_unique<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // MoveIt
  moveit::planning_interface::MoveGroupInterface arm(node, "arm");

  // Safety
  arm.setMaxVelocityScalingFactor(0.1);
  arm.setMaxAccelerationScalingFactor(0.1);
  arm.setPlanningTime(10.0);
  arm.setNumPlanningAttempts(5);

  // Looser tolerances to reduce aborts for 4-DOF arm
  arm.setGoalPositionTolerance(0.03);       // 3cm
  arm.setGoalOrientationTolerance(3.14);    // ignore orientation

  // Gripper action client
  auto gripper = rclcpp_action::create_client<GripperCommand>(node, "/gripper_controller/gripper_cmd");

  // Open gripper at start
  operateGripper(node, gripper, GRIP_OPEN_POS);

  // Search joint waypoints (joint order must match MoveIt group)
  std::vector<std::vector<double>> waypoints = {
    { 0.00, -0.20,  0.20,  0.80},
    { 1.00, -0.20,  0.20,  0.80},
    {-1.00, -0.20,  0.20,  0.80},
    { 0.00, -0.60,  0.30,  1.20}
  };

  int wp_idx = 0;

  // Approach height schedule:
  // target_h is an extra "hover" above detected marker Z. We reduce gradually until final_h.
  double target_h = 0.15;
  double final_h  = 0.035;

  // TF frame names
  const std::string base_frame   = "link1";
  const std::string target_marker = "aruco_marker_23";

  int fail_count = 0;

  // Workspace sanity limits (meters) relative to link1
  // Tweak if your workspace is bigger/smaller.
  const double X_MIN = 0.05, X_MAX = 0.35;
  const double Y_MIN = -0.25, Y_MAX = 0.25;
  const double Z_MIN = 0.00, Z_MAX = 0.50;

  RCLCPP_INFO(node->get_logger(), "=== OMX REAL PICKER (TF marker -> MoveIt position target) ===");
  RCLCPP_INFO(node->get_logger(), "Base frame: %s | Target marker frame: %s", base_frame.c_str(), target_marker.c_str());

  while (rclcpp::ok()) {
    double mx = 0.0, my = 0.0, mz = 0.0;
    bool visible = false;

    // -------- 1) TF lookup --------
    try {
      // Wait a bit for TF to become available
      if (tf_buffer->canTransform(base_frame, target_marker, tf2::TimePointZero, 200ms)) {
        auto t = tf_buffer->lookupTransform(base_frame, target_marker, tf2::TimePointZero);

        // Stamp freshness check (sim time)
        rclcpp::Time now = node->get_clock()->now();
        rclcpp::Time msg_time = t.header.stamp;

        // If stamp is zero (some publishers), skip freshness check
        double delay = 0.0;
        if (msg_time.nanoseconds() > 0) {
          delay = (now - msg_time).seconds();
        }

        if (msg_time.nanoseconds() == 0 || delay < 1.0) {
          mx = t.transform.translation.x;
          my = t.transform.translation.y;
          mz = t.transform.translation.z;

          // Reject insane values early
          if (std::isfinite(mx) && std::isfinite(my) && std::isfinite(mz)) {
            if (mx >= X_MIN && mx <= X_MAX && my >= Y_MIN && my <= Y_MAX && mz >= Z_MIN && mz <= Z_MAX) {
              visible = true;
              RCLCPP_INFO_THROTTLE(
                node->get_logger(), *node->get_clock(), 1000,
                "Marker OK (delay %.2fs) xyz=(%.3f, %.3f, %.3f)", delay, mx, my, mz
              );
            } else {
              RCLCPP_WARN_THROTTLE(
                node->get_logger(), *node->get_clock(), 1000,
                "Marker out of workspace xyz=(%.3f, %.3f, %.3f) -> ignore", mx, my, mz
              );
            }
          }
        } else {
          RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "TF too old (delay %.2fs)", delay);
        }
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000, "TF exception: %s", e.what());
    } catch (...) {
      // ignore
    }

    // -------- 2) Search if not visible or too many fails --------
    if (!visible || fail_count >= 5) {
      if (fail_count >= 5) {
        RCLCPP_WARN(node->get_logger(), ">> Resetting approach due to repeated failures.");
        fail_count = 0;
        target_h = 0.15;
      } else {
        RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 1000, ">> Searching for %s ...", target_marker.c_str());
      }

      // Always refresh start state before planning
      arm.setStartStateToCurrentState();
      arm.clearPoseTargets();

      arm.setJointValueTarget(waypoints[wp_idx]);
      (void)arm.move();  // ignore result; it's a scan motion
      rclcpp::sleep_for(500ms);

      wp_idx = (wp_idx + 1) % waypoints.size();
      continue;
    }

    // -------- 3) Approach using position-only target --------
    if (target_h <= final_h + 0.005) {
      break; // close enough to grasp
    }

    const double target_z = std::max(final_h, mz + target_h);

    RCLCPP_INFO(node->get_logger(), ">> Tracking: hover=%.3f => target_z=%.3f", target_h, target_z);

    // IMPORTANT: refresh MoveIt start state each iteration
    arm.setStartStateToCurrentState();
    arm.clearPoseTargets();

    // Position-only target in base frame (MoveIt will handle orientation as best it can)
    arm.setPositionTarget(mx, my, target_z);

    auto result = arm.move();
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      target_h -= 0.03;
      if (target_h < final_h) target_h = final_h;
      fail_count = 0;
      wp_idx = 0;
    } else {
      fail_count++;
      RCLCPP_ERROR(node->get_logger(), "Move failed (%d/5). Will retry or re-search.", fail_count);
      rclcpp::sleep_for(500ms);
      continue;
    }
  }

  // -------- 4) Grasp --------
  RCLCPP_INFO(node->get_logger(), ">> GRASPING!");
  operateGripper(node, gripper, GRIP_CLOSE_POS);

  // -------- 5) Return home --------
  arm.setStartStateToCurrentState();
  arm.clearPoseTargets();

  arm.setNamedTarget("home");
  (void)arm.move();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
