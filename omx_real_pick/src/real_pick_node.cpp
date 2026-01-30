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
  goal.command.max_effort = 0.5; 
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
   
  // [안전 설정]
  arm.setMaxVelocityScalingFactor(0.1); 
  arm.setMaxAccelerationScalingFactor(0.1);
  arm.setPlanningTime(10.0); // 계산 시간 넉넉히

  // [허용 오차 대폭 증가]
  arm.setGoalPositionTolerance(0.04); // 4cm 오차 허용
  // 4축 로봇 특성상 각도 맞추기가 매우 어려우므로 각도 오차는 무한대로 둡니다.
  arm.setGoalOrientationTolerance(3.14); 

  auto gripper = rclcpp_action::create_client<GripperCommand>(node, "/gripper_controller/gripper_cmd");
  operateGripper(gripper, 0.019); 

  std::vector<std::vector<double>> waypoints = {
    { 0.00, -0.20,  0.20,  0.80},
    { 1.00, -0.20,  0.20,  0.80},
    {-1.00, -0.20,  0.20,  0.80},
    { 0.00, -0.60,  0.30,  1.20}
  };

  int wp_idx = 0;
  double target_h = 0.15; 
  double final_h = 0.035; 

  // [확인] 실제 마커 ID (23번)
  std::string target_marker = "aruco_marker_23"; 
  int fail_count = 0;

  while(rclcpp::ok()) {
    double mx, my, mz;
    bool visible = false;

    // 1. 데이터 조회
    try {
      if(tf_buffer->canTransform("link1", target_marker, tf2::TimePointZero, 100ms)) {
        auto t = tf_buffer->lookupTransform("link1", target_marker, tf2::TimePointZero);
        
        // [수정] 렉이 심하므로 3.0초까지 봐줌 (데이터가 좀 늦게 와도 OK)
        rclcpp::Time now = node->get_clock()->now();
        rclcpp::Time msg_time = t.header.stamp;
        double delay = (now - msg_time).seconds();
        
        if(delay < 3.0) { 
            mx = t.transform.translation.x;
            my = t.transform.translation.y;
            mz = t.transform.translation.z;
            visible = true;
            // [디버깅] 현재 좌표 출력 (너무 이상한 값이면 로봇 못 감)
            RCLCPP_INFO(node->get_logger(), "Marker Found! (Delay: %.2fs) X:%.2f Y:%.2f Z:%.2f", delay, mx, my, mz);
        } else {
            RCLCPP_WARN(node->get_logger(), "Data too old! (Delay: %.2fs)", delay);
        }
      }
    } catch(...) {}

    // 2. 탐색 모드
    if(!visible || fail_count >= 5) { // 5번 실패하면 리셋
      if(fail_count >= 5) {
          RCLCPP_WARN(node->get_logger(), ">> Resetting search...");
          fail_count = 0; 
          target_h = 0.15; 
      } else {
          RCLCPP_INFO(node->get_logger(), ">> Searching... (%s)", target_marker.c_str());
      }
      arm.setJointValueTarget(waypoints[wp_idx]);
      arm.move();
      rclcpp::sleep_for(1s);
      wp_idx = (wp_idx + 1) % waypoints.size();
      continue;
    }

    // 3. 접근 (Position Target 사용)
    RCLCPP_INFO(node->get_logger(), ">> Tracking! H: %.3f", target_h);
    if(target_h <= final_h + 0.005) break; 

    // [핵심 변경] Pose(위치+각도) 대신 Position(위치)만 설정!
    // "손목 각도는 네가 알아서 하고, 제발 이 좌표(x,y,z)로만 가줘"
    arm.setPositionTarget(mx, my, std::max(final_h, mz + target_h));

    auto result = arm.move();

    if(result == moveit::core::MoveItErrorCode::SUCCESS) {
      target_h -= 0.03; 
      if(target_h < final_h) target_h = final_h;
      fail_count = 0; 
      wp_idx = 0; 
    } else {
      fail_count++;
      RCLCPP_ERROR(node->get_logger(), "Move Failed (%d/5). Retrying...", fail_count);
      rclcpp::sleep_for(500ms);
    }
  }

  RCLCPP_INFO(node->get_logger(), ">> GRASPING!");
  operateGripper(gripper, -0.001); 
  arm.setNamedTarget("home");
  arm.move();
  rclcpp::shutdown();
  spinner.join();
  return 0;
}