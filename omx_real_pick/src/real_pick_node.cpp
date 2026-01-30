#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

using GripperCommand = control_msgs::action::GripperCommand;
using namespace std::chrono_literals;

void operateGripper(rclcpp_action::Client<GripperCommand>::SharedPtr client, double pos) {
  if(!client->wait_for_action_server(2s)) return;
  auto goal = GripperCommand::Goal();
  goal.command.position = pos;
  goal.command.max_effort = 0.5; // 실제 로봇은 힘 너무 세게 주면 안됨
  client->async_send_goal(goal);
  rclcpp::sleep_for(1s);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("omx_real_picker");
  auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  exec->add_node(node);
  std::thread spinner([&exec](){ exec->spin(); });

  auto tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  moveit::planning_interface::MoveGroupInterface arm(node, "arm");
  
  // [안전 설정] 실제 로봇은 천천히!
  arm.setMaxVelocityScalingFactor(0.1); 
  arm.setMaxAccelerationScalingFactor(0.1);
  arm.setGoalPositionTolerance(0.015);

  auto gripper = rclcpp_action::create_client<GripperCommand>(node, "/gripper_controller/gripper_cmd");
  operateGripper(gripper, 0.019); // 열기

  // 탐색 웨이포인트 (바닥부터)
  std::vector<std::vector<double>> waypoints = {
    {0.0, 0.20, 0.10, 1.50},   // 발밑
    {0.0, 0.00, 0.20, 1.20},   // 약간 앞
    {0.0, -0.50, 0.40, 1.00}   // 중간
  };

  int wp_idx = 0;
  double target_h = 0.15; // 15cm 상공에서 접근 시작
  double final_h = 0.035; // 잡기 높이 (3.5cm)

  while(rclcpp::ok()) {
    // 1. 마커 확인
    double mx, my, mz;
    bool visible = false;
    try {
      if(tf_buffer->canTransform("link1", "aruco_marker_0", tf2::TimePointZero, 500ms)) {
        auto t = tf_buffer->lookupTransform("link1", "aruco_marker_0", tf2::TimePointZero);
        mx = t.transform.translation.x;
        my = t.transform.translation.y;
        mz = t.transform.translation.z;
        visible = true;
      }
    } catch(...) {}

    if(!visible) {
      RCLCPP_INFO(node->get_logger(), ">> Searching...");
      arm.setJointValueTarget(waypoints[wp_idx]);
      arm.move();
      rclcpp::sleep_for(1s);
      wp_idx = (wp_idx + 1) % waypoints.size();
      continue;
    }

    // 2. 접근 (Spiral Down)
    RCLCPP_INFO(node->get_logger(), ">> Tracking! Dist: %.3f", target_h);
    
    if(target_h <= final_h + 0.005) break; // 도달

    geometry_msgs::msg::Pose p;
    tf2::Quaternion q;
    q.setRPY(0, 0, std::atan2(my, mx));
    p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
    p.position.x = mx; 
    p.position.y = my;
    p.position.z = std::max(final_h, mz + target_h); // 바닥보다 낮게 가지 않도록

    arm.setPoseTarget(p);
    if(arm.move() == moveit::core::MoveItErrorCode::SUCCESS) {
      target_h -= 0.03; // 3cm씩 하강
      if(target_h < final_h) target_h = final_h;
      wp_idx = 0; // 발견하면 탐색 인덱스 초기화
    }
  }

  // 3. 잡기
  RCLCPP_INFO(node->get_logger(), ">> GRASPING!");
  operateGripper(gripper, -0.001); // 닫기
  
  // 4. 들기
  arm.setNamedTarget("home");
  arm.move();
  
  rclcpp::shutdown();
  spinner.join();
  return 0;
}