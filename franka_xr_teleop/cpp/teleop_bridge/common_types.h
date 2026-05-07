#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

namespace teleop {

enum class TeleopState : uint8_t {
  kDisconnected = 0,
  kConnectedIdle = 1,
  kTeleopArmed = 2,
  kTeleopActive = 3,
  kFault = 4,
};

enum class ControlMode : uint8_t {
  kHold = 0,
  kPosition = 1,
  kPose = 2,
};

enum class ControlSource : uint8_t {
  kXr = 0,
  kPolicy = 1,
};

enum class GripperCommandMode : uint8_t {
  kAnalog = 0,
  kBinary = 1,
};

enum class GripperState : uint8_t {
  kOpen = 0,
  kClose = 1,
  kHold = 2,
  kFault = 3,
};

enum class ActionSpace : uint8_t {
  kCartesianDelta = 0,
  kJointPositionAbsolute = 1,
};

inline const char* ToString(ControlMode mode) {
  switch (mode) {
    case ControlMode::kHold:
      return "HOLD";
    case ControlMode::kPosition:
      return "POSITION";
    case ControlMode::kPose:
      return "POSE";
  }
  return "UNKNOWN";
}

inline const char* ToString(ControlSource source) {
  switch (source) {
    case ControlSource::kXr:
      return "XR";
    case ControlSource::kPolicy:
      return "POLICY";
  }
  return "UNKNOWN";
}

inline bool ParseControlMode(std::string_view value, ControlMode* mode) {
  if (value == "position") {
    *mode = ControlMode::kPosition;
    return true;
  }
  if (value == "pose") {
    *mode = ControlMode::kPose;
    return true;
  }
  if (value == "hold") {
    *mode = ControlMode::kHold;
    return true;
  }
  return false;
}

inline bool ParseControlSource(std::string_view value, ControlSource* source) {
  if (value == "xr" || value == "xrobotics" || value == "xrobotics_sdk") {
    *source = ControlSource::kXr;
    return true;
  }
  if (value == "policy" || value == "smolvla") {
    *source = ControlSource::kPolicy;
    return true;
  }
  return false;
}

inline bool ParseGripperCommandMode(std::string_view value, GripperCommandMode* mode) {
  if (value == "analog") {
    *mode = GripperCommandMode::kAnalog;
    return true;
  }
  if (value == "binary") {
    *mode = GripperCommandMode::kBinary;
    return true;
  }
  return false;
}

inline const char* ToString(GripperState state) {
  switch (state) {
    case GripperState::kOpen:
      return "OPEN";
    case GripperState::kClose:
      return "CLOSE";
    case GripperState::kHold:
      return "HOLD";
    case GripperState::kFault:
      return "FAULT";
  }
  return "UNKNOWN";
}

inline const char* ToString(ActionSpace action_space) {
  switch (action_space) {
    case ActionSpace::kCartesianDelta:
      return "cartesian_delta";
    case ActionSpace::kJointPositionAbsolute:
      return "joint_position_absolute";
  }
  return "unknown";
}

struct Pose {
  std::array<double, 3> p{};
  std::array<double, 4> q{{0.0, 0.0, 0.0, 1.0}};  // xyzw
};

struct XRCommand {
  uint64_t timestamp_ns = 0;
  uint64_t sequence_id = 0;
  Pose right_controller_pose{};
  double control_trigger_value = 0.0;
  double gripper_trigger_value = 0.0;
  bool button_a = false;
  bool button_b = false;
  bool right_axis_click = false;
};

struct TeleopAction {
  ActionSpace action_space = ActionSpace::kCartesianDelta;
  std::array<double, 3> delta_translation_m{};
  std::array<double, 3> delta_rotation_rad{};  // angle-axis
  std::array<double, 7> joint_positions_rad{};
  double gripper_command = 0.0;                // normalized gripper state in [0, 1]
};

struct PolicyActionCommand {
  uint64_t timestamp_ns = 0;
  uint64_t sequence_id = 0;
  uint64_t operator_request_id = 0;
  TeleopAction action{};
  bool enabled = false;
  bool episode_start = false;
  bool episode_end = false;
  bool request_rehome = false;
};

struct FaultFlags {
  bool packet_timeout = false;
  bool jump_rejected = false;
  bool workspace_clamped = false;
  bool robot_not_ready = false;
  bool control_exception = false;
  bool ik_rejected = false;
  bool gripper_fault = false;
};

struct RobotSnapshot {
  uint64_t timestamp_ns = 0;
  std::array<double, 7> q{};
  std::array<double, 7> q_d{};
  std::array<double, 7> dq{};
  Pose tcp_pose{};
  Pose tcp_pose_d{};
  std::array<double, 16> F_T_EE{};
  std::array<double, 16> EE_T_K{};
  bool robot_ok = false;
  bool in_reflex = false;
};

struct PlannedTarget {
  uint64_t target_timestamp_ns = 0;
  std::array<double, 7> target_q{};
  Pose desired_tcp_pose{};
  double target_gripper_width_m = 0.0;
  ControlMode control_mode = ControlMode::kHold;
  bool teleop_active = false;
  bool target_fresh = false;
  bool episode_start = false;
  bool episode_end = false;
  double manipulability = 0.0;
  TeleopAction requested_action{};
  TeleopState teleop_state = TeleopState::kDisconnected;
  uint64_t packet_age_ns = 0;
  FaultFlags faults{};
};

struct RobotObservation {
  uint64_t timestamp_ns = 0;
  std::array<double, 7> q{};
  std::array<double, 7> q_cmd{};
  std::array<double, 7> dq{};
  Pose tcp_pose{};
  Pose desired_target_tcp_pose{};
  Pose commanded_target_tcp_pose{};
  double gripper_width = 0.0;
  GripperState gripper_state = GripperState::kOpen;
  TeleopAction executed_action{};
  ControlMode control_mode = ControlMode::kHold;
  TeleopState teleop_state = TeleopState::kDisconnected;
  uint64_t packet_age_ns = 0;
  uint64_t target_age_ns = 0;
  bool target_fresh = false;
  bool teleop_active = false;
  bool episode_start = false;
  bool episode_end = false;
  double target_manipulability = 0.0;
  FaultFlags faults{};
};

struct SafetyLimits {
  double max_translation_speed_mps = 0.20;
  double max_rotation_speed_rps = 0.80;
  double max_step_translation_m = 0.0015;
  double max_step_rotation_rad = 0.010;
  double packet_timeout_s = 0.120;
  double jump_reject_translation_m = 0.08;
  double jump_reject_rotation_rad = 0.80;
  std::array<double, 3> workspace_min{{0.20, -0.45, 0.05}};
  std::array<double, 3> workspace_max{{0.80, 0.45, 0.85}};
  bool enforce_workspace_limits_during_rehome = true;
};

struct TeleopRuntimeConfig {
  ControlMode control_mode = ControlMode::kPose;
  // Mirror the XR client "Switch w/ A Button" option. When enabled, the bridge
  // does not treat A as an episode-start button.
  bool a_button_toggles_robot_control = false;
  // Translation gain: robot_delta_m = scale_factor * xr_delta_m.
  // Values > 1.0 make the robot move farther than your hand (more
  // responsive), < 1.0 attenuate for fine-grained control.
  double scale_factor = 0.8;
  // Rotation gain: robot_rot_delta_rad = rotation_scale_factor * xr_rot_delta_rad.
  // Default 1.0 preserves legacy 1:1 rotation mapping.
  double rotation_scale_factor = 1.0;
  double control_trigger_threshold = 0.9;
  double control_trigger_release_threshold = 0.8;
  double xr_pose_lowpass_alpha = 0.2;
  double xr_translation_deadband_m = 0.0015;
  double xr_rotation_deadband_rad = 0.015;
  double xr_hold_translation_threshold_m = 0.00075;
  double xr_hold_rotation_threshold_rad = 0.010;
  double xr_hold_release_multiplier = 2.5;
  double xr_hold_dwell_s = 0.12;
  double planner_rate_hz = 100.0;
  std::array<double, 7> start_joint_positions_rad{
      {0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483}};
};

struct GripperConfig {
  bool enabled = true;
  GripperCommandMode command_mode = GripperCommandMode::kBinary;
  double max_width_m = 0.08;
  double min_width_m = 0.0;
  double speed_mps = 0.05;
  double min_command_delta_m = 0.002;
  double max_command_rate_hz = 20.0;
  double open_threshold = 0.30;
  double close_threshold = 0.70;
  double toggle_debounce_s = 0.08;
  double stall_width_delta_m = 0.001;
  double stall_timeout_s = 0.25;
  double width_tolerance_m = 0.002;
  double read_failure_timeout_s = 0.50;
  // Force applied when closing via franka::Gripper::grasp(). A modest value
  // is enough to hold light teleop objects without damaging them; the
  // firmware uses this to decide when an object is contacted.
  double grasp_force_n = 20.0;
};

struct IkConfig {
  double damping = 0.05;
  double nullspace_gain = 0.15;
  std::array<double, 7> nullspace_joint_positions_rad{
      {0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483}};
  uint32_t planner_substeps = 1;
  double max_joint_velocity_radps = 0.35;
  double max_joint_acceleration_radps2 = 1.5;
  double max_joint_jerk_radps3 = 12.0;
  double max_joint_step_rad = 0.008;
  double target_smoothing_alpha = 0.25;
  double realtime_target_smoothing_alpha = 0.08;
  double realtime_joint_deadzone_rad = 0.001;
  double realtime_servo_kp = 60.0;
  double realtime_servo_kd = 10.0;
  double realtime_hold_position_threshold_rad = 0.0008;
  double realtime_hold_velocity_threshold_radps = 0.01;
  double realtime_hold_release_threshold_rad = 0.0016;
  double position_gain = 3.0;
  double orientation_gain = 2.0;
  double task_translation_deadband_m = 0.001;
  double task_rotation_deadband_rad = 0.015;
  double manipulability_threshold = 0.05;
  double singularity_damping_gain = 1.0;
};

struct RobotLoadConfig {
  double mass_kg = 0.0;
  std::array<double, 3> center_of_mass_m{{0.0, 0.0, 0.0}};
  std::array<double, 9> inertia_kgm2{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct PolicyRuntimeConfig {
  std::string bind_ip = "0.0.0.0";
  uint16_t action_port = 28082;
  double command_timeout_s = 0.20;
  double max_joint_delta_rad = 0.35;
  double max_joint_distance_rad = 0.60;
  double tracking_position_error_rad = 0.20;
  double tracking_target_error_rad = 0.35;
  double tracking_fault_dwell_s = 0.25;
  double tracking_inhibit_s = 0.75;
};

struct TeleopBridgeConfig {
  SafetyLimits safety{};
  TeleopRuntimeConfig teleop{};
  ControlSource control_source = ControlSource::kXr;
  PolicyRuntimeConfig policy{};
  GripperConfig gripper{};
  IkConfig ik{};
  RobotLoadConfig load{};
  bool limit_rate = true;
  double lpf_cutoff_frequency = 100.0;
  // Position deltas are mapped in the XR/world frame using this rotation.
  std::array<std::array<double, 3>, 3> xr_to_robot_rotation{{
      {{0.0, 0.0, -1.0}},
      {{-1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
  }};
  // Orientation deltas are mapped in the XR/world frame. They can still use
  // different axis signs than translation for intuitive wrist motion, but the
  // command should not depend on the current facing direction of the gripper.
  std::array<std::array<double, 3>, 3> xr_to_robot_rotation_orientation{{
      {{0.0, 0.0, -1.0}},
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
  }};
  bool allow_motion = true;
  std::string robot_ip;
};

struct AppConfig {
  TeleopBridgeConfig bridge{};
  std::string observation_ip = "127.0.0.1";
  uint16_t observation_port = 28081;
  bool dry_run = false;
};

template <typename T>
class LatestValueBuffer {
 public:
  void Publish(const T& value) {
    const uint32_t next = 1u - published_index_.load(std::memory_order_relaxed);
    slots_[next] = value;
    published_index_.store(next, std::memory_order_release);
  }

  T ReadLatest() const {
    const uint32_t idx = published_index_.load(std::memory_order_acquire);
    return slots_[idx];
  }

 private:
  T slots_[2]{};
  std::atomic<uint32_t> published_index_{0};
};

using LatestCommandBuffer = LatestValueBuffer<XRCommand>;
using LatestPolicyActionBuffer = LatestValueBuffer<PolicyActionCommand>;
using LatestObservationBuffer = LatestValueBuffer<RobotObservation>;
using LatestRobotStateBuffer = LatestValueBuffer<RobotSnapshot>;
using LatestPlannedTargetBuffer = LatestValueBuffer<PlannedTarget>;

}  // namespace teleop
