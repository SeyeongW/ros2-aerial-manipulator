#include <cmath>
#include <memory>
#include <string>
#include <atomic>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "aruco_markers_msgs/msg/marker_array.hpp"

#include "rclcpp_action/rclcpp_action.hpp"
#include "control_msgs/action/gripper_command.hpp"

#include "moveit/move_group_interface/move_group_interface.h"

using namespace std::chrono_literals;
using GripperCommand = control_msgs::action::GripperCommand;

class OmxMoveItPickNode : public rclcpp::Node {
public:
  OmxMoveItPickNode()
  : Node("omx_moveit_pick_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Frames / topics
    base_frame_ = declare_parameter<std::string>("base_frame", "link1");
    markers_topic_ = declare_parameter<std::string>("markers_topic", "/aruco/markers");
    target_id_ = static_cast<unsigned int>(declare_parameter<int>("target_id", 0));

    // MoveIt
    planning_group_ = declare_parameter<std::string>("planning_group", "arm");
    ee_link_ = declare_parameter<std::string>("ee_link", "end_effector_link");

    // Distances (m)
    approach_dist_ = declare_parameter<double>("approach_dist", 0.10);
    grasp_advance_ = declare_parameter<double>("grasp_advance", 0.06);
    lift_dist_     = declare_parameter<double>("lift_dist", 0.08);

    // Marker->grasp offset (m): tune these in BASE frame
    offset_x_ = declare_parameter<double>("offset_x", 0.00);
    offset_y_ = declare_parameter<double>("offset_y", 0.00);
    offset_z_ = declare_parameter<double>("offset_z", 0.00);

    // Gripper action
    gripper_action_ = declare_parameter<std::string>("gripper_action", "/gripper_controller/gripper_cmd");
    grip_open_  = declare_parameter<double>("grip_open", 0.010);
    grip_close_ = declare_parameter<double>("grip_close", 0.000);
    grip_effort_= declare_parameter<double>("grip_effort", 2.0);

    // Gripper client
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(this, gripper_action_);

    // Subscriber
    sub_ = create_subscription<aruco_markers_msgs::msg::MarkerArray>(
      markers_topic_, 10, std::bind(&OmxMoveItPickNode::cbMarkers, this, std::placeholders::_1));

    // IMPORTANT: Do NOT construct MoveGroupInterface in constructor.
    // Delay it until the node is fully managed by a shared_ptr.
    init_timer_ = this->create_wall_timer(
      200ms, std::bind(&OmxMoveItPickNode::initMoveIt, this));

    RCLCPP_INFO(get_logger(),
      "Started. markers=%s target_id=%u base=%s group=%s ee=%s",
      markers_topic_.c_str(), target_id_, base_frame_.c_str(),
      planning_group_.c_str(), ee_link_.c_str());
  }

private:
  void initMoveIt()
  {
    if (moveit_ready_) return;

    try {
      // Now shared_from_this() is SAFE because this is called after construction
      auto node_ptr = this->shared_from_this();

      move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
        node_ptr, planning_group_);

      move_group_->setEndEffectorLink(ee_link_);
      move_group_->setPoseReferenceFrame(base_frame_);
      move_group_->setPlanningTime(5.0);
      move_group_->setNumPlanningAttempts(5);

      moveit_ready_ = true;
      RCLCPP_INFO(get_logger(), "MoveIt initialized (MoveGroupInterface ready).");

      // Stop retry timer
      init_timer_->cancel();
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "MoveIt init failed (retrying): %s", e.what());
    }
  }

  void sendGripper(double pos)
  {
    if (!gripper_client_->wait_for_action_server(500ms)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Gripper action server not available.");
      return;
    }
    GripperCommand::Goal goal;
    goal.command.position = pos;
    goal.command.max_effort = grip_effort_;
    gripper_client_->async_send_goal(goal);
  }

  bool planAndExecute(const geometry_msgs::msg::PoseStamped &pose)
  {
    if (!moveit_ready_ || !move_group_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "MoveIt not ready yet.");
      return false;
    }

    move_group_->setPoseTarget(pose, ee_link_);
    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const bool ok = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!ok) {
      RCLCPP_WARN(get_logger(), "Planning failed.");
      move_group_->clearPoseTargets();
      return false;
    }

    const bool ex = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    move_group_->clearPoseTargets();

    if (!ex) {
      RCLCPP_WARN(get_logger(), "Execution failed.");
    }
    return ex;
  }

  bool markerToBasePose(const aruco_markers_msgs::msg::Marker &m, geometry_msgs::msg::PoseStamped &out_base)
  {
    // m.pose is PoseStamped (header.frame_id should be camera frame)
    const auto &in = m.pose;

    // Use latest available transform (TimePointZero) for robustness in sim clocks
    const auto tf = tf_buffer_.lookupTransform(
      base_frame_, in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.2));

    tf2::doTransform(in, out_base, tf);
    out_base.header.frame_id = base_frame_;

    // Apply simple offsets in BASE frame (easy to tune)
    out_base.pose.position.x += offset_x_;
    out_base.pose.position.y += offset_y_;
    out_base.pose.position.z += offset_z_;
    return true;
  }

  void cbMarkers(const aruco_markers_msgs::msg::MarkerArray::SharedPtr msg)
  {
    // If MoveIt isn't ready, don't even start a pick
    if (!moveit_ready_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for MoveIt initialization...");
      return;
    }
    if (busy_) return;

    const aruco_markers_msgs::msg::Marker *chosen = nullptr;
    for (const auto &m : msg->markers) {
      if (m.id == target_id_) { chosen = &m; break; }
    }
    if (!chosen) return;

    busy_ = true;

    try {
      geometry_msgs::msg::PoseStamped target_base;
      markerToBasePose(*chosen, target_base);

      const double x = target_base.pose.position.x;
      const double y = target_base.pose.position.y;
      const double z = target_base.pose.position.z;

      // Approach direction: from robot base to target (radial)
      const double r = std::sqrt(x*x + y*y);
      const double ux = (r > 1e-6) ? (x / r) : 1.0;
      const double uy = (r > 1e-6) ? (y / r) : 0.0;

      geometry_msgs::msg::PoseStamped pre = target_base;
      pre.header.stamp = this->now();
      pre.pose.position.x = x - approach_dist_ * ux;
      pre.pose.position.y = y - approach_dist_ * uy;

      geometry_msgs::msg::PoseStamped grasp = target_base;
      grasp.header.stamp = this->now();
      grasp.pose.position.x = pre.pose.position.x + grasp_advance_ * ux;
      grasp.pose.position.y = pre.pose.position.y + grasp_advance_ * uy;

      geometry_msgs::msg::PoseStamped lift = grasp;
      lift.header.stamp = this->now();
      lift.pose.position.z = z + lift_dist_;

      RCLCPP_INFO(get_logger(), "Pick start: pre->grasp->lift");

      sendGripper(grip_open_);
      rclcpp::sleep_for(500ms);

      if (!planAndExecute(pre))   { busy_ = false; return; }
      if (!planAndExecute(grasp)) { busy_ = false; return; }

      sendGripper(grip_close_);
      rclcpp::sleep_for(700ms);

      planAndExecute(lift);

      RCLCPP_INFO(get_logger(), "Pick done.");

    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "Error: %s", e.what());
    }

    busy_ = false;
  }

  std::atomic<bool> busy_{false};

  // Params
  std::string base_frame_, markers_topic_;
  unsigned target_id_;

  std::string planning_group_, ee_link_;
  double approach_dist_, grasp_advance_, lift_dist_;
  double offset_x_, offset_y_, offset_z_;

  std::string gripper_action_;
  double grip_open_, grip_close_, grip_effort_;

  // ROS
  rclcpp::Subscription<aruco_markers_msgs::msg::MarkerArray>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr init_timer_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // MoveIt (delayed init)
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  bool moveit_ready_{false};

  // Gripper
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OmxMoveItPickNode>());
  rclcpp::shutdown();
  return 0;
}
