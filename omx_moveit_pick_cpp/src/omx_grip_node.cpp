#include <memory>
#include <thread>
#include <vector>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

// tf2::toMsg 대신 직접 값 대입
geometry_msgs::msg::Quaternion createQuaternionFromRPY(double roll, double pitch, double yaw) {
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  geometry_msgs::msg::Quaternion q_msg;
  q_msg.x = q.x(); q_msg.y = q.y(); q_msg.z = q.z(); q_msg.w = q.w();
  return q_msg;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "omx_moveit_pick_place",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  auto executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor->add_node(node);
  std::thread spin_thread([&executor]() { executor->spin(); });

  // 1. MoveIt 그룹 설정 (팔 + 그리퍼)
  static const std::string ARM_GROUP = "arm";
  static const std::string GRIPPER_GROUP = "gripper";

  moveit::planning_interface::MoveGroupInterface move_group_arm(node, ARM_GROUP);
  moveit::planning_interface::MoveGroupInterface move_group_gripper(node, GRIPPER_GROUP);

  // 팔 속도/오차 설정
  move_group_arm.setMaxVelocityScalingFactor(0.5);
  move_group_arm.setMaxAccelerationScalingFactor(0.5);
  move_group_arm.setGoalPositionTolerance(0.02);   
  move_group_arm.setGoalOrientationTolerance(0.1);

  // ★ 그리퍼 오차 설정 (MoveIt 명령이 씹히는 것 방지)
  move_group_gripper.setMaxVelocityScalingFactor(1.0);
  move_group_gripper.setGoalPositionTolerance(0.005); // 민감도 조절

  RCLCPP_INFO(node->get_logger(), "Planning Frame: %s", move_group_arm.getPlanningFrame().c_str());
  move_group_arm.setStartStateToCurrentState();

  // === 좌표 설정 ===
  double pick_x = 0.200;
  double pick_y = 0.000;
  double pick_z = 0.014; // 1.5cm
  double approach_z_offset = 0.050;

  // ---------------------------------------------------------
  // [Step 1] 홈 이동
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Moving to HOME");
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();
  rclcpp::sleep_for(std::chrono::seconds(1));

  // ---------------------------------------------------------
  // [Step 2] 그리퍼 열기 (MoveIt Named Target 사용)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Gripper OPEN (MoveIt)");
  // SRDF에 정의된 "open" 상태를 사용 (없으면 에러 로그 뜸)
  if (move_group_gripper.setNamedTarget("open")) {
      move_group_gripper.move();
  } else {
      // Named Target이 없으면 직접 값 입력
      move_group_gripper.setJointValueTarget({0.019});
      move_group_gripper.move();
  }
  rclcpp::sleep_for(std::chrono::seconds(1)); // 물리적 이동 대기

  // ---------------------------------------------------------
  // [Step 3] Pick 접근 (Above)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Moving ABOVE Pick");
  geometry_msgs::msg::Pose target_pose;
  double yaw_pick = std::atan2(pick_y, pick_x);
  target_pose.orientation = createQuaternionFromRPY(0.0, 0.0, yaw_pick);
  target_pose.position.x = pick_x;
  target_pose.position.y = pick_y;
  target_pose.position.z = pick_z + approach_z_offset;

  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  // ---------------------------------------------------------
  // [Step 4] 내려가기 (Down)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Moving DOWN to Pick");
  target_pose.position.z = pick_z;
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  // ---------------------------------------------------------
  // [Step 5] 잡기 (MoveIt Named Target 사용)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> GRASPING (MoveIt)");
  // "close" 대신 "grip"이라고 되어 있을 수도 있음. 실패시 수동 값 사용
  if (move_group_gripper.setNamedTarget("close")) {
      move_group_gripper.move();
  } else {
      // 강제로 꽉 잡기 위해 음수 값 시도 (MoveIt은 경고 띄울 수 있음)
      // 안전하게 0.000 (완전 닫힘) 근처 값 사용
      move_group_gripper.setJointValueTarget({-0.005}); 
      move_group_gripper.move();
  }
  rclcpp::sleep_for(std::chrono::seconds(1));

  // ---------------------------------------------------------
  // [Step 6] 들어올리기 (Lift)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> LIFTING");
  target_pose.position.z = pick_z + approach_z_offset;
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  // ---------------------------------------------------------
  // [Step 7] 놓을 위치로 이동 (Place) - 관절 제어
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Moving to Place Location");
  std::vector<double> place_joint_positions = {1.5, -0.6, 0.3, 0.7}; 
  move_group_arm.setJointValueTarget(place_joint_positions);
  move_group_arm.move();

  // ---------------------------------------------------------
  // [Step 8] 그리퍼 열기 (MoveIt)
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> RELEASING (MoveIt)");
  if (move_group_gripper.setNamedTarget("open")) {
      move_group_gripper.move();
  } else {
      move_group_gripper.setJointValueTarget({0.019});
      move_group_gripper.move();
  }
  rclcpp::sleep_for(std::chrono::seconds(1));

  // ---------------------------------------------------------
  // [Step 9] 홈 복귀
  // ---------------------------------------------------------
  RCLCPP_INFO(node->get_logger(), ">> Returning HOME");
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  RCLCPP_INFO(node->get_logger(), "Mission Complete!");
  
  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}