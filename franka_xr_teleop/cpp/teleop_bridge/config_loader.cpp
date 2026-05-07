#include "config_loader.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace teleop {
namespace {

template <typename T>
bool ReadScalar(const YAML::Node& node, const char* key, T* out) {
  if (!node[key]) {
    return false;
  }
  *out = node[key].as<T>();
  return true;
}

template <size_t N>
bool ReadArray(const YAML::Node& node, const char* key, std::array<double, N>* out) {
  if (!node[key] || !node[key].IsSequence() || node[key].size() != N) {
    return false;
  }
  for (size_t i = 0; i < N; ++i) {
    (*out)[i] = node[key][i].as<double>();
  }
  return true;
}

bool ReadMatrix3(const YAML::Node& node,
                 const char* key,
                 std::array<std::array<double, 3>, 3>* out) {
  if (!node[key] || !node[key].IsSequence() || node[key].size() != 3) {
    return false;
  }
  for (size_t row = 0; row < 3; ++row) {
    if (!node[key][row].IsSequence() || node[key][row].size() != 3) {
      return false;
    }
    for (size_t col = 0; col < 3; ++col) {
      (*out)[row][col] = node[key][row][col].as<double>();
    }
  }
  return true;
}

std::string JoinPath(const std::string& root, const std::string& leaf) {
  return (std::filesystem::path(root) / leaf).string();
}

bool LoadRobotConfig(const std::string& path, AppConfig* config, std::string* error) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node robot = root["robot"];
  if (!robot || !robot.IsMap()) {
    *error = "Missing 'robot' map in " + path;
    return false;
  }
  ReadScalar(robot, "ip", &config->bridge.robot_ip);
  ReadScalar(robot, "load_mass_kg", &config->bridge.load.mass_kg);
  ReadArray(robot, "load_center_of_mass_m", &config->bridge.load.center_of_mass_m);
  ReadArray(robot, "load_inertia_kgm2", &config->bridge.load.inertia_kgm2);
  ReadScalar(robot, "limit_rate", &config->bridge.limit_rate);
  ReadScalar(robot, "lpf_cutoff_frequency", &config->bridge.lpf_cutoff_frequency);
  return true;
}

bool LoadSafetyConfig(const std::string& path, AppConfig* config, std::string* error) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node safety = root["safety"];
  if (!safety || !safety.IsMap()) {
    *error = "Missing 'safety' map in " + path;
    return false;
  }
  ReadScalar(safety, "packet_timeout_s", &config->bridge.safety.packet_timeout_s);
  ReadScalar(safety, "max_translation_speed_mps", &config->bridge.safety.max_translation_speed_mps);
  ReadScalar(safety, "max_rotation_speed_rps", &config->bridge.safety.max_rotation_speed_rps);
  ReadScalar(safety, "max_step_translation_m", &config->bridge.safety.max_step_translation_m);
  ReadScalar(safety, "max_step_rotation_rad", &config->bridge.safety.max_step_rotation_rad);
  ReadScalar(safety, "jump_reject_translation_m", &config->bridge.safety.jump_reject_translation_m);
  ReadScalar(safety, "jump_reject_rotation_rad", &config->bridge.safety.jump_reject_rotation_rad);
  ReadArray(safety, "workspace_min_xyz", &config->bridge.safety.workspace_min);
  ReadArray(safety, "workspace_max_xyz", &config->bridge.safety.workspace_max);
  ReadScalar(safety,
             "enforce_workspace_limits_during_rehome",
             &config->bridge.safety.enforce_workspace_limits_during_rehome);
  return true;
}

bool LoadTeleopConfig(const std::string& path, AppConfig* config, std::string* error) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node teleop = root["teleop"];
  if (!teleop || !teleop.IsMap()) {
    *error = "Missing 'teleop' map in " + path;
    return false;
  }

  ReadScalar(teleop, "observation_ip", &config->observation_ip);
  int observation_port = static_cast<int>(config->observation_port);
  if (ReadScalar(teleop, "observation_port", &observation_port)) {
    config->observation_port = static_cast<uint16_t>(observation_port);
  }
  ReadScalar(teleop, "allow_motion", &config->bridge.allow_motion);
  ReadScalar(teleop, "dry_run", &config->dry_run);
  std::string control_source;
  if (ReadScalar(teleop, "control_source", &control_source) ||
      ReadScalar(teleop, "input_source", &control_source)) {
    if (!ParseControlSource(control_source, &config->bridge.control_source)) {
      *error = "Unsupported teleop.control_source '" + control_source + "' in " + path;
      return false;
    }
  }
  ReadScalar(teleop,
             "a_button_toggles_robot_control",
             &config->bridge.teleop.a_button_toggles_robot_control);
  ReadScalar(teleop, "scale_factor", &config->bridge.teleop.scale_factor);
  ReadScalar(teleop, "rotation_scale_factor", &config->bridge.teleop.rotation_scale_factor);
  ReadScalar(teleop, "control_trigger_threshold", &config->bridge.teleop.control_trigger_threshold);
  ReadScalar(teleop,
             "control_trigger_release_threshold",
             &config->bridge.teleop.control_trigger_release_threshold);
  ReadScalar(teleop, "xr_pose_lowpass_alpha", &config->bridge.teleop.xr_pose_lowpass_alpha);
  ReadScalar(teleop, "xr_translation_deadband_m", &config->bridge.teleop.xr_translation_deadband_m);
  ReadScalar(teleop, "xr_rotation_deadband_rad", &config->bridge.teleop.xr_rotation_deadband_rad);
  ReadScalar(teleop,
             "xr_hold_translation_threshold_m",
             &config->bridge.teleop.xr_hold_translation_threshold_m);
  ReadScalar(teleop,
             "xr_hold_rotation_threshold_rad",
             &config->bridge.teleop.xr_hold_rotation_threshold_rad);
  ReadScalar(teleop,
             "xr_hold_release_multiplier",
             &config->bridge.teleop.xr_hold_release_multiplier);
  ReadScalar(teleop, "xr_hold_dwell_s", &config->bridge.teleop.xr_hold_dwell_s);
  ReadScalar(teleop, "planner_rate_hz", &config->bridge.teleop.planner_rate_hz);
  ReadArray(teleop, "start_joint_positions_rad", &config->bridge.teleop.start_joint_positions_rad);

  std::string control_mode;
  if (ReadScalar(teleop, "control_mode", &control_mode)) {
    if (!ParseControlMode(control_mode, &config->bridge.teleop.control_mode) ||
        config->bridge.teleop.control_mode == ControlMode::kHold) {
      *error = "Unsupported teleop.control_mode '" + control_mode + "' in " + path;
      return false;
    }
  }

  if (const YAML::Node ik = teleop["ik"]; ik && ik.IsMap()) {
    ReadScalar(ik, "damping", &config->bridge.ik.damping);
    ReadScalar(ik, "nullspace_gain", &config->bridge.ik.nullspace_gain);
    ReadArray(ik,
              "nullspace_joint_positions_rad",
              &config->bridge.ik.nullspace_joint_positions_rad);
    ReadScalar(ik, "planner_substeps", &config->bridge.ik.planner_substeps);
    ReadScalar(ik, "max_joint_velocity_radps", &config->bridge.ik.max_joint_velocity_radps);
    ReadScalar(ik, "max_joint_acceleration_radps2", &config->bridge.ik.max_joint_acceleration_radps2);
    ReadScalar(ik, "max_joint_jerk_radps3", &config->bridge.ik.max_joint_jerk_radps3);
    ReadScalar(ik, "max_joint_step_rad", &config->bridge.ik.max_joint_step_rad);
    ReadScalar(ik, "target_smoothing_alpha", &config->bridge.ik.target_smoothing_alpha);
    ReadScalar(ik,
               "realtime_target_smoothing_alpha",
               &config->bridge.ik.realtime_target_smoothing_alpha);
    ReadScalar(ik, "realtime_joint_deadzone_rad", &config->bridge.ik.realtime_joint_deadzone_rad);
    ReadScalar(ik, "realtime_servo_kp", &config->bridge.ik.realtime_servo_kp);
    ReadScalar(ik, "realtime_servo_kd", &config->bridge.ik.realtime_servo_kd);
    ReadScalar(ik,
               "realtime_hold_position_threshold_rad",
               &config->bridge.ik.realtime_hold_position_threshold_rad);
    ReadScalar(ik,
               "realtime_hold_velocity_threshold_radps",
               &config->bridge.ik.realtime_hold_velocity_threshold_radps);
    ReadScalar(ik,
               "realtime_hold_release_threshold_rad",
               &config->bridge.ik.realtime_hold_release_threshold_rad);
    ReadScalar(ik, "position_gain", &config->bridge.ik.position_gain);
    ReadScalar(ik, "orientation_gain", &config->bridge.ik.orientation_gain);
    ReadScalar(ik, "task_translation_deadband_m", &config->bridge.ik.task_translation_deadband_m);
    ReadScalar(ik, "task_rotation_deadband_rad", &config->bridge.ik.task_rotation_deadband_rad);
    ReadScalar(ik, "manipulability_threshold", &config->bridge.ik.manipulability_threshold);
    ReadScalar(ik, "singularity_damping_gain", &config->bridge.ik.singularity_damping_gain);
  }

  if (const YAML::Node gripper = teleop["gripper"]; gripper && gripper.IsMap()) {
    ReadScalar(gripper, "enabled", &config->bridge.gripper.enabled);
    std::string command_mode;
    if (ReadScalar(gripper, "command_mode", &command_mode) &&
        !ParseGripperCommandMode(command_mode, &config->bridge.gripper.command_mode)) {
      *error = "Unsupported teleop.gripper.command_mode '" + command_mode + "' in " + path;
      return false;
    }
    ReadScalar(gripper, "max_width_m", &config->bridge.gripper.max_width_m);
    ReadScalar(gripper, "min_width_m", &config->bridge.gripper.min_width_m);
    ReadScalar(gripper, "speed_mps", &config->bridge.gripper.speed_mps);
    ReadScalar(gripper, "min_command_delta_m", &config->bridge.gripper.min_command_delta_m);
    ReadScalar(gripper, "max_command_rate_hz", &config->bridge.gripper.max_command_rate_hz);
    ReadScalar(gripper, "open_threshold", &config->bridge.gripper.open_threshold);
    ReadScalar(gripper, "close_threshold", &config->bridge.gripper.close_threshold);
    ReadScalar(gripper, "toggle_debounce_s", &config->bridge.gripper.toggle_debounce_s);
    ReadScalar(gripper, "stall_width_delta_m", &config->bridge.gripper.stall_width_delta_m);
    ReadScalar(gripper, "stall_timeout_s", &config->bridge.gripper.stall_timeout_s);
    ReadScalar(gripper, "width_tolerance_m", &config->bridge.gripper.width_tolerance_m);
    ReadScalar(gripper, "read_failure_timeout_s", &config->bridge.gripper.read_failure_timeout_s);
    ReadScalar(gripper, "grasp_force_n", &config->bridge.gripper.grasp_force_n);
  }

  if (const YAML::Node policy = teleop["policy"]; policy && policy.IsMap()) {
    ReadScalar(policy, "bind_ip", &config->bridge.policy.bind_ip);
    int action_port = static_cast<int>(config->bridge.policy.action_port);
    if (ReadScalar(policy, "action_port", &action_port)) {
      config->bridge.policy.action_port = static_cast<uint16_t>(action_port);
    }
    ReadScalar(policy, "command_timeout_s", &config->bridge.policy.command_timeout_s);
    ReadScalar(policy, "max_joint_delta_rad", &config->bridge.policy.max_joint_delta_rad);
    ReadScalar(policy, "max_joint_distance_rad", &config->bridge.policy.max_joint_distance_rad);
    ReadScalar(policy, "tracking_position_error_rad", &config->bridge.policy.tracking_position_error_rad);
    ReadScalar(policy, "tracking_target_error_rad", &config->bridge.policy.tracking_target_error_rad);
    ReadScalar(policy, "tracking_fault_dwell_s", &config->bridge.policy.tracking_fault_dwell_s);
    ReadScalar(policy, "tracking_inhibit_s", &config->bridge.policy.tracking_inhibit_s);
  }

  return true;
}

bool LoadXrFrameConfig(const std::string& path, AppConfig* config, std::string* error) {
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node xr_frame = root["xr_frame"];
  if (!xr_frame || !xr_frame.IsMap()) {
    *error = "Missing 'xr_frame' map in " + path;
    return false;
  }
  if (!ReadMatrix3(xr_frame, "rotation_matrix_row_major", &config->bridge.xr_to_robot_rotation)) {
    *error = "Missing/invalid xr_frame.rotation_matrix_row_major in " + path;
    return false;
  }
  ReadMatrix3(xr_frame,
              "orientation_rotation_matrix_row_major",
              &config->bridge.xr_to_robot_rotation_orientation);
  return true;
}

}  // namespace

bool LoadAppConfig(const std::string& config_dir, AppConfig* config, std::string* error) {
  try {
    if (!LoadRobotConfig(JoinPath(config_dir, "robot.yaml"), config, error)) {
      return false;
    }
    if (!LoadSafetyConfig(JoinPath(config_dir, "safety.yaml"), config, error)) {
      return false;
    }
    if (!LoadTeleopConfig(JoinPath(config_dir, "teleop.yaml"), config, error)) {
      return false;
    }
    if (!LoadXrFrameConfig(JoinPath(config_dir, "xr_frame.yaml"), config, error)) {
      return false;
    }
  } catch (const std::exception& e) {
    *error = std::string("Failed to load config: ") + e.what();
    return false;
  }

  if (config->bridge.robot_ip.empty()) {
    *error = "robot.ip is required";
    return false;
  }
  if (config->bridge.teleop.planner_rate_hz <= 1.0) {
    *error = "teleop.planner_rate_hz must be > 1.0";
    return false;
  }
  if (config->bridge.policy.action_port == 0) {
    *error = "teleop.policy.action_port must be in [1, 65535]";
    return false;
  }
  if (config->bridge.policy.command_timeout_s <= 0.0) {
    *error = "teleop.policy.command_timeout_s must be > 0";
    return false;
  }
  if (config->bridge.policy.max_joint_delta_rad <= 0.0) {
    *error = "teleop.policy.max_joint_delta_rad must be > 0";
    return false;
  }
  if (config->bridge.policy.max_joint_distance_rad <= 0.0) {
    *error = "teleop.policy.max_joint_distance_rad must be > 0";
    return false;
  }
  if (config->bridge.policy.tracking_position_error_rad <= 0.0) {
    *error = "teleop.policy.tracking_position_error_rad must be > 0";
    return false;
  }
  if (config->bridge.policy.tracking_target_error_rad <= 0.0) {
    *error = "teleop.policy.tracking_target_error_rad must be > 0";
    return false;
  }
  if (config->bridge.policy.tracking_fault_dwell_s <= 0.0) {
    *error = "teleop.policy.tracking_fault_dwell_s must be > 0";
    return false;
  }
  if (config->bridge.policy.tracking_inhibit_s <= 0.0) {
    *error = "teleop.policy.tracking_inhibit_s must be > 0";
    return false;
  }
  if (config->bridge.teleop.control_trigger_threshold < 0.0 ||
      config->bridge.teleop.control_trigger_threshold > 1.0) {
    *error = "teleop.control_trigger_threshold must be in [0, 1]";
    return false;
  }
  if (config->bridge.teleop.control_trigger_release_threshold < 0.0 ||
      config->bridge.teleop.control_trigger_release_threshold > 1.0) {
    *error = "teleop.control_trigger_release_threshold must be in [0, 1]";
    return false;
  }
  if (config->bridge.teleop.control_trigger_release_threshold >=
      config->bridge.teleop.control_trigger_threshold) {
    *error = "teleop.control_trigger_release_threshold must be < teleop.control_trigger_threshold";
    return false;
  }
  if (config->bridge.teleop.xr_pose_lowpass_alpha < 0.0 ||
      config->bridge.teleop.xr_pose_lowpass_alpha > 1.0) {
    *error = "teleop.xr_pose_lowpass_alpha must be in [0, 1]";
    return false;
  }
  if (config->bridge.teleop.xr_translation_deadband_m < 0.0) {
    *error = "teleop.xr_translation_deadband_m must be >= 0";
    return false;
  }
  if (config->bridge.teleop.xr_rotation_deadband_rad < 0.0) {
    *error = "teleop.xr_rotation_deadband_rad must be >= 0";
    return false;
  }
  if (config->bridge.teleop.xr_hold_translation_threshold_m < 0.0) {
    *error = "teleop.xr_hold_translation_threshold_m must be >= 0";
    return false;
  }
  if (config->bridge.teleop.xr_hold_rotation_threshold_rad < 0.0) {
    *error = "teleop.xr_hold_rotation_threshold_rad must be >= 0";
    return false;
  }
  if (config->bridge.teleop.xr_hold_release_multiplier < 1.0) {
    *error = "teleop.xr_hold_release_multiplier must be >= 1.0";
    return false;
  }
  if (config->bridge.teleop.xr_hold_dwell_s < 0.0) {
    *error = "teleop.xr_hold_dwell_s must be >= 0";
    return false;
  }
  if (config->bridge.load.mass_kg < 0.0) {
    *error = "robot.load_mass_kg must be >= 0";
    return false;
  }
  if (config->bridge.lpf_cutoff_frequency <= 0.0) {
    *error = "robot.lpf_cutoff_frequency must be > 0";
    return false;
  }
  if (config->bridge.gripper.min_width_m < 0.0 ||
      config->bridge.gripper.max_width_m < config->bridge.gripper.min_width_m) {
    *error = "Invalid gripper width range";
    return false;
  }
  if (config->bridge.gripper.min_command_delta_m < 0.0) {
    *error = "teleop.gripper.min_command_delta_m must be >= 0";
    return false;
  }
  if (config->bridge.gripper.max_command_rate_hz <= 0.0) {
    *error = "teleop.gripper.max_command_rate_hz must be > 0";
    return false;
  }
  if (config->bridge.gripper.open_threshold < 0.0 ||
      config->bridge.gripper.open_threshold > 1.0) {
    *error = "teleop.gripper.open_threshold must be in [0, 1]";
    return false;
  }
  if (config->bridge.gripper.close_threshold < 0.0 ||
      config->bridge.gripper.close_threshold > 1.0) {
    *error = "teleop.gripper.close_threshold must be in [0, 1]";
    return false;
  }
  if (config->bridge.gripper.open_threshold > config->bridge.gripper.close_threshold) {
    *error = "teleop.gripper.open_threshold must be <= teleop.gripper.close_threshold";
    return false;
  }
  if (config->bridge.gripper.toggle_debounce_s < 0.0) {
    *error = "teleop.gripper.toggle_debounce_s must be >= 0";
    return false;
  }
  if (config->bridge.gripper.stall_width_delta_m < 0.0) {
    *error = "teleop.gripper.stall_width_delta_m must be >= 0";
    return false;
  }
  if (config->bridge.gripper.stall_timeout_s < 0.0) {
    *error = "teleop.gripper.stall_timeout_s must be >= 0";
    return false;
  }
  if (config->bridge.gripper.width_tolerance_m < 0.0) {
    *error = "teleop.gripper.width_tolerance_m must be >= 0";
    return false;
  }
  if (config->bridge.gripper.read_failure_timeout_s < 0.0) {
    *error = "teleop.gripper.read_failure_timeout_s must be >= 0";
    return false;
  }
  if (config->bridge.safety.max_translation_speed_mps <= 0.0) {
    *error = "safety.max_translation_speed_mps must be > 0";
    return false;
  }
  if (config->bridge.safety.max_rotation_speed_rps <= 0.0) {
    *error = "safety.max_rotation_speed_rps must be > 0";
    return false;
  }
  if (config->bridge.safety.max_step_translation_m <= 0.0) {
    *error = "safety.max_step_translation_m must be > 0";
    return false;
  }
  if (config->bridge.safety.max_step_rotation_rad <= 0.0) {
    *error = "safety.max_step_rotation_rad must be > 0";
    return false;
  }
  if (config->bridge.safety.packet_timeout_s <= 0.0) {
    *error = "safety.packet_timeout_s must be > 0";
    return false;
  }
  if (config->bridge.safety.jump_reject_translation_m <= 0.0) {
    *error = "safety.jump_reject_translation_m must be > 0";
    return false;
  }
  if (config->bridge.safety.jump_reject_rotation_rad <= 0.0) {
    *error = "safety.jump_reject_rotation_rad must be > 0";
    return false;
  }
  for (size_t i = 0; i < config->bridge.safety.workspace_min.size(); ++i) {
    if (config->bridge.safety.workspace_min[i] > config->bridge.safety.workspace_max[i]) {
      *error = "safety.workspace_min_xyz must be <= safety.workspace_max_xyz on every axis";
      return false;
    }
  }
  if (config->bridge.ik.max_joint_acceleration_radps2 <= 0.0) {
    *error = "teleop.ik.max_joint_acceleration_radps2 must be > 0";
    return false;
  }
  if (config->bridge.ik.max_joint_jerk_radps3 <= 0.0) {
    *error = "teleop.ik.max_joint_jerk_radps3 must be > 0";
    return false;
  }
  if (config->bridge.ik.planner_substeps == 0) {
    *error = "teleop.ik.planner_substeps must be >= 1";
    return false;
  }
  if (config->bridge.ik.target_smoothing_alpha < 0.0 ||
      config->bridge.ik.target_smoothing_alpha > 1.0) {
    *error = "teleop.ik.target_smoothing_alpha must be in [0, 1]";
    return false;
  }
  if (config->bridge.ik.realtime_target_smoothing_alpha < 0.0 ||
      config->bridge.ik.realtime_target_smoothing_alpha > 1.0) {
    *error = "teleop.ik.realtime_target_smoothing_alpha must be in [0, 1]";
    return false;
  }
  if (config->bridge.ik.realtime_joint_deadzone_rad < 0.0) {
    *error = "teleop.ik.realtime_joint_deadzone_rad must be >= 0";
    return false;
  }
  if (config->bridge.ik.realtime_servo_kp <= 0.0) {
    *error = "teleop.ik.realtime_servo_kp must be > 0";
    return false;
  }
  if (config->bridge.ik.realtime_servo_kd < 0.0) {
    *error = "teleop.ik.realtime_servo_kd must be >= 0";
    return false;
  }
  if (config->bridge.ik.realtime_hold_position_threshold_rad < 0.0) {
    *error = "teleop.ik.realtime_hold_position_threshold_rad must be >= 0";
    return false;
  }
  if (config->bridge.ik.realtime_hold_velocity_threshold_radps < 0.0) {
    *error = "teleop.ik.realtime_hold_velocity_threshold_radps must be >= 0";
    return false;
  }
  if (config->bridge.ik.realtime_hold_release_threshold_rad <
      config->bridge.ik.realtime_hold_position_threshold_rad) {
    *error =
        "teleop.ik.realtime_hold_release_threshold_rad must be >= teleop.ik.realtime_hold_position_threshold_rad";
    return false;
  }
  if (config->bridge.ik.task_translation_deadband_m < 0.0) {
    *error = "teleop.ik.task_translation_deadband_m must be >= 0";
    return false;
  }
  if (config->bridge.ik.task_rotation_deadband_rad < 0.0) {
    *error = "teleop.ik.task_rotation_deadband_rad must be >= 0";
    return false;
  }
  return true;
}

bool SaveStartJointPositions(const std::string& config_dir,
                             const std::array<double, 7>& start_joint_positions_rad,
                             std::string* error) {
  const std::string path = JoinPath(config_dir, "teleop.yaml");

  try {
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node teleop = root["teleop"];
    if (!teleop || !teleop.IsMap()) {
      *error = "Missing 'teleop' map in " + path;
      return false;
    }

    YAML::Node start_joint_positions(YAML::NodeType::Sequence);
    start_joint_positions.SetStyle(YAML::EmitterStyle::Flow);
    for (const double joint_position : start_joint_positions_rad) {
      start_joint_positions.push_back(joint_position);
    }
    teleop["start_joint_positions_rad"] = start_joint_positions;

    YAML::Emitter emitter;
    emitter.SetIndent(2);
    emitter << root;
    if (!emitter.good()) {
      *error = "Failed to emit YAML for " + path;
      return false;
    }

    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) {
      *error = "Failed to open " + path + " for writing";
      return false;
    }

    output << emitter.c_str() << "\n";
    if (!output.good()) {
      *error = "Failed to write " + path;
      return false;
    }
  } catch (const std::exception& e) {
    *error = std::string("Failed to save start joints: ") + e.what();
    return false;
  }

  return true;
}

}  // namespace teleop
