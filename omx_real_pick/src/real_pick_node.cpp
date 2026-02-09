#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <control_msgs/action/gripper_command.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/time.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <deque>
#include <algorithm>
#include <mutex>

using GripperCommand = control_msgs::action::GripperCommand;
using namespace std::chrono_literals;

// ------------------------------
// 유틸
// ------------------------------
static double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

static bool isFinite(double v) {
  return std::isfinite(v);
}

// ------------------------------
// 그리퍼 동작(간단 버전)
// - async goal만 날리고 잠깐 sleep
// ------------------------------
void operateGripper(rclcpp::Node::SharedPtr node,
                    rclcpp_action::Client<GripperCommand>::SharedPtr client,
                    double pos)
{
  if (!client->wait_for_action_server(2s)) {
    RCLCPP_ERROR(node->get_logger(), "Gripper action server not available");
    return;
  }
  auto goal = GripperCommand::Goal();
  goal.command.position = pos;
  goal.command.max_effort = 0.5;
  client->async_send_goal(goal);
  rclcpp::sleep_for(800ms);
}

// ------------------------------
// FSM 상태 정의
// ------------------------------
enum class FSM {
  SEARCH,        // waypoint로 탐색하며 마커 보이길 기다림
  STABILIZE,     // N번 연속으로 신선한 pose가 들어오면 안정화 완료
  APPROACH,      // step 방식으로 점진 접근
  GRASP,         // 그리퍼 닫기
  LIFT_AND_HOME  // 들어올리고 home 복귀
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("omx_real_picker");
  auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  exec->add_node(node);
  std::thread spinner([&exec](){ exec->spin(); });

  // ---------------------------------
  // TF: PoseStamped(카메라 프레임) -> link1 변환에 필요
  // ---------------------------------
  auto tf_buffer = std::make_unique<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // ---------------------------------
  // MoveIt
  // ---------------------------------
  moveit::planning_interface::MoveGroupInterface arm(node, "arm");
  arm.setMaxVelocityScalingFactor(0.15);
  arm.setMaxAccelerationScalingFactor(0.15);
  arm.setPlanningTime(5.0);

  // 4DOF라 orientation은 사실상 무시하고 position 위주
  arm.setGoalPositionTolerance(0.03);      // 3cm
  arm.setGoalOrientationTolerance(3.14);   // 사실상 무시

  // ---------------------------------
  // Gripper Action
  // ---------------------------------
  auto gripper =
    rclcpp_action::create_client<GripperCommand>(node, "/gripper_controller/gripper_cmd");

  // ---------- USER SETTINGS ----------
  // base_frame: 로봇 제어 기준 프레임 (MoveIt에서 position target 기준)
  std::string base_frame = "link1";

  // 탐색용 waypoint들
  std::vector<std::vector<double>> waypoints = {
    { 0.00, -0.20,  0.20,  0.80},
    { 1.00, -0.20,  0.20,  0.80},
    {-1.00, -0.20,  0.20,  0.80},
    { 0.00, -0.60,  0.30,  1.20}
  };

  // Gripper positions
  double GRIP_OPEN  = 0.019;
  double GRIP_CLOSE = -0.001;

  // APPROACH params
  double hover_offset = 0.15;     // 마커 위/앞에서 시작할 거리
  double step = 0.03;             // 접근 step
  double final_min_z = 0.04;      // 바닥 꽂힘 방지용 최소 z (base_frame 기준)
  double max_tf_age_sec = 0.7;    // pose/TF 신선도 컷
  int    stable_need = 5;         // 연속 N번 성공 시 stabilize 완료
  int    max_fail_before_search = 3;

  // filtering window (median)
  const size_t FILTER_N = 7;
  std::deque<double> fx, fy, fz;

  auto push_and_median = [&](std::deque<double>& dq, double v) -> double {
    dq.push_back(v);
    if (dq.size() > FILTER_N) dq.pop_front();
    std::vector<double> tmp(dq.begin(), dq.end());
    std::sort(tmp.begin(), tmp.end());
    return tmp[tmp.size()/2];
  };

  // ---------------------------------
  // (핵심) Python에서 넘어오는 신호/포즈 저장소
  // - 멀티스레드(스피너)에서 콜백으로 갱신되므로 mutex로 보호
  // ---------------------------------
  geometry_msgs::msg::PoseStamped last_pose_cam;
  rclcpp::Time last_pose_time(0, 0, RCL_ROS_TIME);
  bool pose_valid = false;
  bool target_visible = false;
  std::mutex mtx;

  // ---------------------------------
  // 구독 1) 타겟 visible 신호
  // ---------------------------------
  auto sub_vis = node->create_subscription<std_msgs::msg::Bool>(
    "/aruco/target_visible", 10,
    [&](const std_msgs::msg::Bool::SharedPtr msg){
      std::lock_guard<std::mutex> lk(mtx);
      target_visible = msg->data;
    }
  );

  // ---------------------------------
  // 구독 2) 타겟 PoseStamped (카메라 프레임 기준)
  // ---------------------------------
  auto sub_pose = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/aruco/target_pose", 10,
    [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg){
      std::lock_guard<std::mutex> lk(mtx);
      last_pose_cam = *msg;
      last_pose_time = msg->header.stamp;
      pose_valid = true;
    }
  );

  // ---------------------------------
  // PoseStamped 기반으로 "신선한 마커(=타겟) 좌표"를 얻는 함수
  //
  // 절차:
  // 1) target_visible && pose_valid 확인
  // 2) age 체크 (max_tf_age_sec)
  // 3) cam_pose를 base_frame(link1)로 TF 변환
  // 4) median filter 적용 후 ox,oy,oz 출력
  // ---------------------------------
  auto getFreshMarkerFromTopic = [&](double& ox, double& oy, double& oz) -> bool {
    geometry_msgs::msg::PoseStamped cam_pose;
    rclcpp::Time stamp(0,0,RCL_ROS_TIME);
    bool vis=false, valid=false;

    {
      std::lock_guard<std::mutex> lk(mtx);
      vis = target_visible;
      valid = pose_valid;
      cam_pose = last_pose_cam;
      stamp = last_pose_time;
    }

    if (!vis || !valid) return false;

    // 신선도 체크
    rclcpp::Time now = node->get_clock()->now();
    double age = (now - stamp).seconds();
    if (age > max_tf_age_sec) {
      RCLCPP_WARN(node->get_logger(), "Target pose too old: %.2fs (ignore)", age);
      return false;
    }

    // frame_id가 없으면 변환 자체가 불가
    if (cam_pose.header.frame_id.empty()) return false;

    // cam -> base_frame 변환
    geometry_msgs::msg::PoseStamped base_pose;
    try {
      base_pose = tf_buffer->transform(cam_pose, base_frame, tf2::durationFromSec(0.05));
    } catch (...) {
      return false;
    }

    double x = base_pose.pose.position.x;
    double y = base_pose.pose.position.y;
    double z = base_pose.pose.position.z;

    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) return false;

    // median filtering
    ox = push_and_median(fx, x);
    oy = push_and_median(fy, y);
    oz = push_and_median(fz, z);
    return true;
  };

  // ---------- INIT ----------
  RCLCPP_INFO(node->get_logger(), ">> HOME");
  arm.setNamedTarget("home");
  arm.move();
  rclcpp::sleep_for(800ms);

  operateGripper(node, gripper, GRIP_OPEN);

  FSM state = FSM::SEARCH;
  int wp_idx = 0;
  int stable_count = 0;
  int approach_fail = 0;

  // 접근 높이: hover_offset에서 시작해서 step으로 감소
  double current_offset = hover_offset;

  // filtered target position (link1 기준)
  double mx=0, my=0, mz=0;

  // ---------- LOOP ----------
  rclcpp::Rate rate(10);

  while (rclcpp::ok()) {

    // (핵심) 이제 visible은 "topic + transform"으로 판단한다.
    bool visible = getFreshMarkerFromTopic(mx, my, mz);

    switch (state) {

      case FSM::SEARCH: {
        // SEARCH에선 안정화 카운트 초기화 + 필터 초기화
        stable_count = 0;
        fx.clear(); fy.clear(); fz.clear();

        if (visible) {
          RCLCPP_INFO(node->get_logger(), ">> Target seen. Switching to STABILIZE");
          state = FSM::STABILIZE;
          break;
        }

        // waypoint 기반 탐색
        RCLCPP_INFO(node->get_logger(), ">> SEARCH waypoint %d", wp_idx);
        arm.setJointValueTarget(waypoints[wp_idx]);
        arm.move();

        wp_idx = (wp_idx + 1) % waypoints.size();
        rclcpp::sleep_for(300ms);
        break;
      }

      case FSM::STABILIZE: {
        // 타겟 잃으면 다시 SEARCH
        if (!visible) {
          stable_count = 0;
          RCLCPP_WARN(node->get_logger(), ">> Lost target. Back to SEARCH");
          state = FSM::SEARCH;
          break;
        }

        // 연속으로 들어오는지 체크
        stable_count++;
        RCLCPP_INFO(node->get_logger(),
                    ">> STABILIZE %d/%d  (x=%.3f y=%.3f z=%.3f)",
                    stable_count, stable_need, mx, my, mz);

        if (stable_count >= stable_need) {
          // 접근 초기화
          current_offset = hover_offset;
          approach_fail = 0;

          RCLCPP_INFO(node->get_logger(), ">> STABILIZE done. Switching to APPROACH");
          state = FSM::APPROACH;
        }
        break;
      }

      case FSM::APPROACH: {
        // 접근 중에도 타겟 잃으면 STABILIZE로
        if (!visible) {
          RCLCPP_WARN(node->get_logger(), ">> Lost target during approach. Back to STABILIZE");
          stable_count = 0;
          state = FSM::STABILIZE;
          break;
        }

        // 목표 z = (타겟 z + current_offset), 단 최저 높이 보호
        double target_z = std::max(final_min_z, mz + current_offset);

        RCLCPP_INFO(node->get_logger()),
                    ">> APPROACH offset=%.3f  target=(%.3f, %.3f, %.3f)",
