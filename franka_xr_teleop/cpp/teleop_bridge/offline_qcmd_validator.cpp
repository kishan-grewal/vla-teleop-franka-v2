#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <nlohmann/json.hpp>

#include "common_types.h"
#include "config_loader.h"
#include "math_utils.h"

namespace teleop {
namespace {

constexpr std::array<double, 7> kPandaJointLowerLimitsRad{
    {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
constexpr std::array<double, 7> kPandaJointUpperLimitsRad{
    {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
constexpr double kJointLimitMarginRad = 0.02;
constexpr double kSegmentGapThresholdS = 0.1;
constexpr size_t kSegmentSeedIkIterations = 10;
constexpr double kSegmentSeedPoseToleranceM = 1e-3;
constexpr double kSegmentSeedRotationToleranceRad = 5e-3;

struct Options {
  std::string config_dir = "configs";
  std::string input_path;
  std::string output_path;
  std::string robot_ip_override;
  std::string method = "commanded_pose_ik";
  size_t skip_rows = 0;
  size_t max_rows = 0;
  bool write_output = false;
};

struct ObservationRow {
  nlohmann::json raw;
  uint64_t timestamp_ns = 0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::optional<std::array<double, 7>> q_cmd_truth;
  Pose tcp_pose{};
  Pose desired_target_pose{};
  Pose commanded_target_pose{};
  ControlMode control_mode = ControlMode::kHold;
  bool teleop_active = false;
  bool target_fresh = false;
  FaultFlags faults{};
};

struct MetricsAccumulator {
  size_t count = 0;
  double mae_sum = 0.0;
  double max_abs_error = 0.0;
  std::array<double, 7> mae_joint_sum{};
  std::array<double, 7> max_abs_joint{};
  double pose_position_error_sum = 0.0;
  double pose_position_error_max = 0.0;
  double pose_rotation_error_sum = 0.0;
  double pose_rotation_error_max = 0.0;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " --input PATH [options]\n"
      << "Options:\n"
      << "  --config-dir DIR         Config directory (default: configs)\n"
      << "  --robot-ip IP           Override robot IP from config\n"
      << "  --method NAME           replay_desired | commanded_pose_ik (default: commanded_pose_ik)\n"
      << "  --output PATH           Write augmented JSONL with backfilled_q_cmd\n"
      << "  --skip-rows N           Skip the first N parsed rows (default: 0)\n"
      << "  --max-rows N            Stop after N rows (default: all)\n";
}

bool ParseSize(const std::string& value, size_t* out) {
  try {
    *out = static_cast<size_t>(std::stoull(value));
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* flag, std::string* out) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      *out = argv[++i];
      return true;
    };

    if (arg == "--input") {
      if (!require_value("--input", &options->input_path)) {
        return false;
      }
    } else if (arg == "--output") {
      if (!require_value("--output", &options->output_path)) {
        return false;
      }
      options->write_output = true;
    } else if (arg == "--config-dir") {
      if (!require_value("--config-dir", &options->config_dir)) {
        return false;
      }
    } else if (arg == "--robot-ip") {
      if (!require_value("--robot-ip", &options->robot_ip_override)) {
        return false;
      }
    } else if (arg == "--method") {
      if (!require_value("--method", &options->method)) {
        return false;
      }
    } else if (arg == "--max-rows") {
      std::string value;
      if (!require_value("--max-rows", &value)) {
        return false;
      }
      if (!ParseSize(value, &options->max_rows)) {
        std::cerr << "Invalid integer for --max-rows: " << value << "\n";
        return false;
      }
    } else if (arg == "--skip-rows") {
      std::string value;
      if (!require_value("--skip-rows", &value)) {
        return false;
      }
      if (!ParseSize(value, &options->skip_rows)) {
        std::cerr << "Invalid integer for --skip-rows: " << value << "\n";
        return false;
      }
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options->input_path.empty()) {
    std::cerr << "--input is required\n";
    return false;
  }
  if (options->method != "replay_desired" && options->method != "commanded_pose_ik") {
    std::cerr << "Unsupported --method: " << options->method << "\n";
    return false;
  }
  return true;
}

template <size_t N>
bool ParseNumberArray(const nlohmann::json& value, std::array<double, N>* out) {
  if (!value.is_array() || value.size() != N) {
    return false;
  }
  for (size_t i = 0; i < N; ++i) {
    if (!value[i].is_number()) {
      return false;
    }
    (*out)[i] = value[i].get<double>();
  }
  return true;
}

bool ParsePose(const nlohmann::json& value, Pose* pose) {
  if (!value.is_object()) {
    return false;
  }
  return ParseNumberArray<3>(value.at("tcp_position_xyz"), &pose->p) &&
         ParseNumberArray<4>(value.at("tcp_orientation_xyzw"), &pose->q);
}

bool ParseControlModeJson(const nlohmann::json& value, ControlMode* mode) {
  if (!value.is_string()) {
    return false;
  }
  std::string text = value.get<std::string>();
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ParseControlMode(text, mode);
}

bool ParseFaultFlags(const nlohmann::json& value, FaultFlags* faults) {
  if (!value.is_object()) {
    return false;
  }
  auto read = [&](const char* key, bool* out) {
    if (value.contains(key) && value.at(key).is_boolean()) {
      *out = value.at(key).get<bool>();
    }
  };
  read("packet_timeout", &faults->packet_timeout);
  read("jump_rejected", &faults->jump_rejected);
  read("workspace_clamped", &faults->workspace_clamped);
  read("robot_not_ready", &faults->robot_not_ready);
  read("control_exception", &faults->control_exception);
  read("ik_rejected", &faults->ik_rejected);
  read("gripper_fault", &faults->gripper_fault);
  return true;
}

bool ParseObservationRow(const nlohmann::json& root, ObservationRow* row, std::string* error) {
  try {
    row->raw = root;
    if (!root.contains("timestamp_ns") || !root.at("timestamp_ns").is_number_integer()) {
      *error = "missing timestamp_ns";
      return false;
    }
    row->timestamp_ns = root.at("timestamp_ns").get<uint64_t>();

    const nlohmann::json& robot_state = root.at("robot_state");
    const nlohmann::json& desired_target_state = root.at("desired_target_state");
    const nlohmann::json& commanded_target_state = root.at("commanded_target_state");
    const nlohmann::json& status = root.at("status");

    if (!ParseNumberArray<7>(robot_state.at("q"), &row->q)) {
      *error = "invalid robot_state.q";
      return false;
    }
    if (!ParseNumberArray<7>(robot_state.at("dq"), &row->dq)) {
      *error = "invalid robot_state.dq";
      return false;
    }
    if (!ParsePose(robot_state, &row->tcp_pose)) {
      *error = "invalid robot_state tcp pose";
      return false;
    }
    if (!ParsePose(desired_target_state, &row->desired_target_pose)) {
      *error = "invalid desired_target_state pose";
      return false;
    }
    if (!ParsePose(commanded_target_state, &row->commanded_target_pose)) {
      *error = "invalid commanded_target_state pose";
      return false;
    }
    if (!ParseControlModeJson(status.at("control_mode"), &row->control_mode)) {
      *error = "invalid status.control_mode";
      return false;
    }
    if (robot_state.contains("q_cmd")) {
      std::array<double, 7> q_cmd{};
      if (ParseNumberArray<7>(robot_state.at("q_cmd"), &q_cmd)) {
        row->q_cmd_truth = q_cmd;
      }
    }
    if (status.contains("teleop_active") && status.at("teleop_active").is_boolean()) {
      row->teleop_active = status.at("teleop_active").get<bool>();
    }
    if (status.contains("target_fresh") && status.at("target_fresh").is_boolean()) {
      row->target_fresh = status.at("target_fresh").get<bool>();
    }
    if (status.contains("fault_flags")) {
      ParseFaultFlags(status.at("fault_flags"), &row->faults);
    }
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
  return true;
}

std::array<double, 7> ClampToJointLimits(const std::array<double, 7>& q) {
  std::array<double, 7> out = q;
  for (size_t i = 0; i < 7; ++i) {
    const double lo = kPandaJointLowerLimitsRad[i] + kJointLimitMarginRad;
    const double hi = kPandaJointUpperLimitsRad[i] - kJointLimitMarginRad;
    out[i] = std::clamp(out[i], lo, hi);
  }
  return out;
}

Eigen::Matrix<double, 6, 7> JacobianToEigen(const std::array<double, 42>& jacobian_col_major) {
  return Eigen::Map<const Eigen::Matrix<double, 6, 7, Eigen::ColMajor>>(jacobian_col_major.data());
}

double ComputeManipulability(const Eigen::Matrix<double, 6, 7>& jacobian) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(
      jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd singular_values = svd.singularValues();
  double product = 1.0;
  for (int i = 0; i < singular_values.size(); ++i) {
    product *= std::max(singular_values[i], 0.0);
  }
  return product;
}

Eigen::Vector3d ApplyVectorDeadband(const Eigen::Vector3d& value, double deadband) {
  const double norm = value.norm();
  if (norm <= deadband || norm <= 1e-12) {
    return Eigen::Vector3d::Zero();
  }
  return value * ((norm - deadband) / norm);
}

bool SolveIkStep(const franka::Model& model,
                 const RobotSnapshot& snapshot,
                 const Pose& desired_pose,
                 const TeleopBridgeConfig& config,
                 ControlMode control_mode,
                 std::array<double, 7>* target_q,
                 double* manipulability) {
  if (control_mode == ControlMode::kHold) {
    *target_q = snapshot.q_d;
    *manipulability = 0.0;
    return true;
  }

  std::array<double, 7> q_next = snapshot.q_d;
  Pose tcp_pose_next = snapshot.tcp_pose_d;
  const Eigen::Matrix<double, 7, 1> q_nullspace_target =
      Eigen::Map<const Eigen::Matrix<double, 7, 1>>(config.ik.nullspace_joint_positions_rad.data());
  const double dt = 1.0 / config.teleop.planner_rate_hz;
  const double max_step_by_velocity = config.ik.max_joint_velocity_radps * dt;
  const double max_step = std::min(config.ik.max_joint_step_rad, max_step_by_velocity);
  *manipulability = 0.0;

  for (uint32_t substep = 0; substep < config.ik.planner_substeps; ++substep) {
    const Eigen::Matrix<double, 6, 7> jacobian = JacobianToEigen(
        model.zeroJacobian(franka::Frame::kEndEffector, q_next, snapshot.F_T_EE, snapshot.EE_T_K));
    Eigen::Matrix<double, 6, 7> jacobian_task = jacobian;

    Eigen::Matrix<double, 6, 1> task_error = Eigen::Matrix<double, 6, 1>::Zero();
    const Eigen::Vector3d raw_position_error = ToEigen(desired_pose.p) - ToEigen(tcp_pose_next.p);
    const Eigen::Vector3d position_error =
        ApplyVectorDeadband(raw_position_error, config.ik.task_translation_deadband_m);
    task_error.head<3>() = config.ik.position_gain * position_error;

    if (control_mode == ControlMode::kPose) {
      const Eigen::Vector3d raw_rotation_error =
          QuaternionErrorAngleAxis(ToEigenQuat(tcp_pose_next.q), ToEigenQuat(desired_pose.q));
      const Eigen::Vector3d rotation_error =
          ApplyVectorDeadband(raw_rotation_error, config.ik.task_rotation_deadband_rad);
      task_error.tail<3>() = config.ik.orientation_gain * rotation_error;
    } else {
      jacobian_task.bottomRows<3>().setZero();
    }

    *manipulability = ComputeManipulability(jacobian_task);

    double lambda = config.ik.damping;
    if (*manipulability < config.ik.manipulability_threshold) {
      lambda +=
          (config.ik.manipulability_threshold - *manipulability) * config.ik.singularity_damping_gain;
    }

    const Eigen::Matrix<double, 6, 6> a_matrix =
        jacobian_task * jacobian_task.transpose() +
        (lambda * lambda) * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> a_ldlt(a_matrix);
    if (a_ldlt.info() != Eigen::Success) {
      return false;
    }

    const Eigen::Matrix<double, 7, 1> q_current =
        Eigen::Map<const Eigen::Matrix<double, 7, 1>>(q_next.data());
    const Eigen::Matrix<double, 7, 1> dq_primary =
        jacobian_task.transpose() * a_ldlt.solve(task_error);
    const Eigen::Matrix<double, 7, 7> nullspace_projector =
        Eigen::Matrix<double, 7, 7>::Identity() -
        jacobian_task.transpose() * a_ldlt.solve(jacobian_task);
    const Eigen::Matrix<double, 7, 1> dq =
        dq_primary + config.ik.nullspace_gain * nullspace_projector * (q_nullspace_target - q_current);

    for (size_t i = 0; i < 7; ++i) {
      if (!std::isfinite(dq[i])) {
        return false;
      }
      const double step = std::clamp(dq[i] * dt, -max_step, max_step);
      q_next[i] += step;
    }
    q_next = ClampToJointLimits(q_next);
    tcp_pose_next = MatrixToPose(
        model.pose(franka::Frame::kEndEffector, q_next, snapshot.F_T_EE, snapshot.EE_T_K));
  }

  *target_q = ClampToJointLimits(q_next);
  return true;
}

class JointPositionTrajectoryGenerator {
 public:
  explicit JointPositionTrajectoryGenerator(const TeleopBridgeConfig& config)
      : max_velocity_(std::max(config.ik.max_joint_velocity_radps, 1e-6)),
        max_acceleration_(std::max(config.ik.max_joint_acceleration_radps2, 1e-6)),
        max_jerk_(std::max(config.ik.max_joint_jerk_radps3, 1e-6)),
        max_step_(std::max(0.0, config.ik.max_joint_step_rad)),
        joint_deadzone_(std::max(0.0, config.ik.realtime_joint_deadzone_rad)),
        servo_kp_(std::max(config.ik.realtime_servo_kp, 1e-6)),
        servo_kd_(std::max(0.0, config.ik.realtime_servo_kd)),
        hold_position_threshold_(std::max(0.0, config.ik.realtime_hold_position_threshold_rad)),
        hold_velocity_threshold_(
            std::max(0.0, config.ik.realtime_hold_velocity_threshold_radps)),
        hold_release_threshold_(
            std::max(config.ik.realtime_hold_position_threshold_rad,
                     config.ik.realtime_hold_release_threshold_rad)) {}

  void Reset() {
    initialized_ = false;
    motion_active_ = false;
    hold_active_ = false;
    hold_target_q_.fill(0.0);
    q_ref_.fill(0.0);
    dq_ref_.fill(0.0);
    ddq_ref_.fill(0.0);
  }

  std::array<double, 7> Update(const std::array<double, 7>& target_q,
                               const std::array<double, 7>& reference_q,
                               double dt,
                               bool apply_motion) {
    if (!initialized_) {
      q_ref_ = reference_q;
      dq_ref_.fill(0.0);
      ddq_ref_.fill(0.0);
      hold_target_q_ = q_ref_;
      initialized_ = true;
    }

    if (!apply_motion) {
      if (motion_active_) {
        hold_target_q_ = reference_q;
      }
      motion_active_ = false;
    } else {
      motion_active_ = true;
      hold_target_q_ = target_q;
    }

    dt = std::max(dt, 1e-6);
    const double max_step_from_velocity = max_velocity_ * dt;
    const double max_step = max_step_ > 0.0 ? std::min(max_step_, max_step_from_velocity)
                                            : max_step_from_velocity;
    const double max_delta_acceleration = max_jerk_ * dt;

    if (!apply_motion) {
      std::array<double, 7> q_out = q_ref_;
      std::array<double, 7> dq_out = dq_ref_;
      std::array<double, 7> ddq_out = ddq_ref_;

      for (size_t i = 0; i < 7; ++i) {
        const double desired_acceleration =
            std::clamp(-hold_velocity_damping_ * dq_ref_[i], -max_acceleration_, max_acceleration_);
        ddq_out[i] = ddq_ref_[i] +
                     std::clamp(desired_acceleration - ddq_ref_[i],
                                -max_delta_acceleration,
                                max_delta_acceleration);
        ddq_out[i] = std::clamp(ddq_out[i], -max_acceleration_, max_acceleration_);

        const double velocity_unclamped = dq_ref_[i] + ddq_out[i] * dt;
        dq_out[i] = std::clamp(velocity_unclamped, -max_velocity_, max_velocity_);
        if ((dq_ref_[i] > 0.0 && dq_out[i] < 0.0) || (dq_ref_[i] < 0.0 && dq_out[i] > 0.0)) {
          dq_out[i] = 0.0;
          ddq_out[i] = 0.0;
        }

        const double step_unclamped = dq_out[i] * dt;
        const double step = std::clamp(step_unclamped, -max_step, max_step);
        q_out[i] = q_ref_[i] + step;
        const double joint_lo = kPandaJointLowerLimitsRad[i] + kJointLimitMarginRad;
        const double joint_hi = kPandaJointUpperLimitsRad[i] - kJointLimitMarginRad;
        const double clamped_q_out = std::clamp(q_out[i], joint_lo, joint_hi);
        if (std::abs(clamped_q_out - q_out[i]) > 1e-12) {
          q_out[i] = clamped_q_out;
          dq_out[i] = 0.0;
          ddq_out[i] = 0.0;
        }
      }

      q_ref_ = q_out;
      dq_ref_ = dq_out;
      ddq_ref_ = ddq_out;
      hold_target_q_ = q_ref_;
      return q_out;
    }

    const std::array<double, 7>& active_target_q = target_q;
    double max_abs_hold_error = 0.0;
    bool within_hold_band = true;
    for (size_t i = 0; i < 7; ++i) {
      const double error_to_ref = active_target_q[i] - q_ref_[i];
      max_abs_hold_error = std::max(max_abs_hold_error, std::abs(error_to_ref));
      if (std::abs(error_to_ref) > hold_position_threshold_ ||
          std::abs(dq_ref_[i]) > hold_velocity_threshold_) {
        within_hold_band = false;
      }
    }

    if (hold_active_ && max_abs_hold_error <= hold_release_threshold_) {
      dq_ref_.fill(0.0);
      ddq_ref_.fill(0.0);
      return q_ref_;
    }

    if (within_hold_band) {
      hold_active_ = true;
      dq_ref_.fill(0.0);
      ddq_ref_.fill(0.0);
      return q_ref_;
    }

    hold_active_ = false;
    std::array<double, 7> q_out = q_ref_;
    std::array<double, 7> dq_out = dq_ref_;
    std::array<double, 7> ddq_out = ddq_ref_;

    for (size_t i = 0; i < 7; ++i) {
      double error = active_target_q[i] - q_ref_[i];
      if (std::abs(error) < joint_deadzone_) {
        error = 0.0;
      }

      const double desired_acceleration =
          std::clamp(servo_kp_ * error - servo_kd_ * dq_ref_[i],
                     -max_acceleration_,
                     max_acceleration_);
      ddq_out[i] = ddq_ref_[i] +
                   std::clamp(desired_acceleration - ddq_ref_[i],
                              -max_delta_acceleration,
                              max_delta_acceleration);
      ddq_out[i] = std::clamp(ddq_out[i], -max_acceleration_, max_acceleration_);

      const double velocity_unclamped = dq_ref_[i] + ddq_out[i] * dt;
      dq_out[i] = std::clamp(velocity_unclamped, -max_velocity_, max_velocity_);

      const double step_unclamped = dq_out[i] * dt;
      const double step = std::clamp(step_unclamped, -max_step, max_step);
      q_out[i] = q_ref_[i] + step;
      const double joint_lo = kPandaJointLowerLimitsRad[i] + kJointLimitMarginRad;
      const double joint_hi = kPandaJointUpperLimitsRad[i] - kJointLimitMarginRad;
      const double clamped_q_out = std::clamp(q_out[i], joint_lo, joint_hi);
      if (std::abs(clamped_q_out - q_out[i]) > 1e-12) {
        q_out[i] = clamped_q_out;
        dq_out[i] = 0.0;
        ddq_out[i] = 0.0;
      }

      if (error != 0.0) {
        const double new_error = active_target_q[i] - q_out[i];
        if ((error > 0.0 && new_error < 0.0) || (error < 0.0 && new_error > 0.0)) {
          q_out[i] = active_target_q[i];
          dq_out[i] = 0.0;
          ddq_out[i] = 0.0;
        }
      }
    }

    q_ref_ = q_out;
    dq_ref_ = dq_out;
    ddq_ref_ = ddq_out;
    return q_out;
  }

 private:
  double max_velocity_ = 0.35;
  double max_acceleration_ = 1.5;
  double max_jerk_ = 12.0;
  double max_step_ = 0.008;
  double joint_deadzone_ = 0.001;
  double servo_kp_ = 60.0;
  double servo_kd_ = 10.0;
  double hold_velocity_damping_ = 12.0;
  double hold_position_threshold_ = 0.0008;
  double hold_velocity_threshold_ = 0.01;
  double hold_release_threshold_ = 0.0016;
  bool initialized_ = false;
  bool motion_active_ = false;
  bool hold_active_ = false;
  std::array<double, 7> hold_target_q_{};
  std::array<double, 7> q_ref_{};
  std::array<double, 7> dq_ref_{};
  std::array<double, 7> ddq_ref_{};
};

Pose ForwardPose(const franka::Model& model,
                 const std::array<double, 7>& q,
                 const std::array<double, 16>& F_T_EE,
                 const std::array<double, 16>& EE_T_K) {
  return MatrixToPose(model.pose(franka::Frame::kEndEffector, q, F_T_EE, EE_T_K));
}

double PositionErrorNorm(const Pose& a, const Pose& b) {
  return (ToEigen(a.p) - ToEigen(b.p)).norm();
}

double RotationErrorNorm(const Pose& a, const Pose& b) {
  return QuaternionErrorAngleAxis(ToEigenQuat(a.q), ToEigenQuat(b.q)).norm();
}

bool ShouldApplyMotion(const ObservationRow& row) {
  return row.teleop_active && row.target_fresh && row.control_mode != ControlMode::kHold &&
         !row.faults.packet_timeout && !row.faults.jump_rejected && !row.faults.workspace_clamped &&
         !row.faults.robot_not_ready && !row.faults.ik_rejected && !row.faults.control_exception;
}

bool IsSegmentStart(const ObservationRow& row, const ObservationRow* previous_row) {
  if (previous_row == nullptr) {
    return true;
  }
  if (row.timestamp_ns <= previous_row->timestamp_ns) {
    return true;
  }
  const double dt = static_cast<double>(row.timestamp_ns - previous_row->timestamp_ns) * 1e-9;
  if (dt > kSegmentGapThresholdS) {
    return true;
  }

  const bool previous_apply_motion = ShouldApplyMotion(*previous_row);
  const bool current_apply_motion = ShouldApplyMotion(row);
  if (!previous_apply_motion && current_apply_motion) {
    return true;
  }
  if (!previous_row->teleop_active && row.teleop_active) {
    return true;
  }
  if (!previous_row->target_fresh && row.target_fresh) {
    return true;
  }
  if (previous_row->control_mode == ControlMode::kHold && row.control_mode != ControlMode::kHold) {
    return true;
  }
  return false;
}

std::array<double, 7> ComputeSegmentSeed(const ObservationRow& row,
                                         const franka::Model& model,
                                         const TeleopBridgeConfig& config,
                                         const std::array<double, 16>& F_T_EE,
                                         const std::array<double, 16>& EE_T_K) {
  std::array<double, 7> q_seed = ClampToJointLimits(row.q);
  if (!ShouldApplyMotion(row)) {
    return q_seed;
  }

  for (size_t iter = 0; iter < kSegmentSeedIkIterations; ++iter) {
    RobotSnapshot snapshot{};
    snapshot.q = row.q;
    snapshot.q_d = q_seed;
    snapshot.dq = row.dq;
    snapshot.tcp_pose = row.tcp_pose;
    snapshot.tcp_pose_d = ForwardPose(model, q_seed, F_T_EE, EE_T_K);
    snapshot.F_T_EE = F_T_EE;
    snapshot.EE_T_K = EE_T_K;
    snapshot.robot_ok = true;

    std::array<double, 7> q_next = q_seed;
    double manipulability = 0.0;
    const bool ik_ok = SolveIkStep(model,
                                   snapshot,
                                   row.commanded_target_pose,
                                   config,
                                   row.control_mode,
                                   &q_next,
                                   &manipulability);
    if (!ik_ok) {
      break;
    }
    q_seed = q_next;

    const Pose seeded_pose = ForwardPose(model, q_seed, F_T_EE, EE_T_K);
    if (PositionErrorNorm(seeded_pose, row.commanded_target_pose) <= kSegmentSeedPoseToleranceM &&
        RotationErrorNorm(seeded_pose, row.commanded_target_pose) <=
            kSegmentSeedRotationToleranceRad) {
      break;
    }
  }

  return q_seed;
}

std::array<double, 7> RunReplayDesired(const ObservationRow& row,
                                       const ObservationRow* previous_row,
                                       const franka::Model& model,
                                       const TeleopBridgeConfig& config,
                                       const std::array<double, 16>& F_T_EE,
                                       const std::array<double, 16>& EE_T_K,
                                       JointPositionTrajectoryGenerator* trajectory_generator,
                                       bool segment_start,
                                       std::array<double, 7>* previous_q_cmd) {
  const std::array<double, 7> q_d =
      (previous_row == nullptr || segment_start) ? *previous_q_cmd : *previous_q_cmd;
  const Pose tcp_pose_d =
      (previous_row == nullptr || segment_start) ? ForwardPose(model, q_d, F_T_EE, EE_T_K)
                                                 : ForwardPose(model, q_d, F_T_EE, EE_T_K);

  RobotSnapshot snapshot{};
  snapshot.q = row.q;
  snapshot.q_d = q_d;
  snapshot.dq = row.dq;
  snapshot.tcp_pose = row.tcp_pose;
  snapshot.tcp_pose_d = tcp_pose_d;
  snapshot.F_T_EE = F_T_EE;
  snapshot.EE_T_K = EE_T_K;
  snapshot.robot_ok = !row.faults.robot_not_ready;

  std::array<double, 7> planned_target = q_d;
  double manipulability = 0.0;
  const bool ik_ok = SolveIkStep(model,
                                 snapshot,
                                 row.desired_target_pose,
                                 config,
                                 row.control_mode,
                                 &planned_target,
                                 &manipulability);
  if (!ik_ok) {
    planned_target = q_d;
  }

  double dt = 1.0 / config.teleop.planner_rate_hz;
  if (!segment_start && previous_row != nullptr && row.timestamp_ns > previous_row->timestamp_ns) {
    dt = static_cast<double>(row.timestamp_ns - previous_row->timestamp_ns) * 1e-9;
  }
  const bool apply_motion = ShouldApplyMotion(row) && ik_ok;
  std::array<double, 7> q_cmd = trajectory_generator->Update(planned_target, q_d, dt, apply_motion);
  *previous_q_cmd = q_cmd;
  return q_cmd;
}

std::array<double, 7> RunCommandedPoseIk(const ObservationRow& row,
                                         const ObservationRow* previous_row,
                                         const franka::Model& model,
                                         const TeleopBridgeConfig& config,
                                         const std::array<double, 16>& F_T_EE,
                                         const std::array<double, 16>& EE_T_K,
                                         std::array<double, 7>* previous_q_cmd) {
  std::array<double, 7> q_est = previous_row == nullptr ? row.q : *previous_q_cmd;
  q_est = ClampToJointLimits(q_est);

  constexpr size_t kMaxIterations = 64;
  for (size_t iter = 0; iter < kMaxIterations; ++iter) {
    RobotSnapshot snapshot{};
    snapshot.q = row.q;
    snapshot.q_d = q_est;
    snapshot.dq = row.dq;
    snapshot.tcp_pose = row.tcp_pose;
    snapshot.tcp_pose_d = ForwardPose(model, q_est, F_T_EE, EE_T_K);
    snapshot.F_T_EE = F_T_EE;
    snapshot.EE_T_K = EE_T_K;
    snapshot.robot_ok = true;

    std::array<double, 7> q_next = q_est;
    double manipulability = 0.0;
    const bool ik_ok = SolveIkStep(model,
                                   snapshot,
                                   row.commanded_target_pose,
                                   config,
                                   ControlMode::kPose,
                                   &q_next,
                                   &manipulability);
    if (!ik_ok) {
      break;
    }

    const Pose next_pose = ForwardPose(model, q_next, F_T_EE, EE_T_K);
    const double pos_error = PositionErrorNorm(next_pose, row.commanded_target_pose);
    const double rot_error = RotationErrorNorm(next_pose, row.commanded_target_pose);
    double max_joint_step = 0.0;
    for (size_t joint = 0; joint < 7; ++joint) {
      max_joint_step = std::max(max_joint_step, std::abs(q_next[joint] - q_est[joint]));
    }
    q_est = q_next;
    if (pos_error <= 1e-4 && rot_error <= 1e-3 && max_joint_step <= 1e-4) {
      break;
    }
  }

  *previous_q_cmd = q_est;
  return q_est;
}

std::array<double, 7> ComputeEstimate(const ObservationRow& row,
                                      const ObservationRow* previous_row,
                                      const franka::Model& model,
                                      const TeleopBridgeConfig& config,
                                      const std::array<double, 16>& F_T_EE,
                                      const std::array<double, 16>& EE_T_K,
                                      const std::string& method,
                                      JointPositionTrajectoryGenerator* trajectory_generator,
                                      bool segment_start,
                                      std::array<double, 7>* previous_q_cmd) {
  if (method == "replay_desired") {
    return RunReplayDesired(row,
                            previous_row,
                            model,
                            config,
                            F_T_EE,
                            EE_T_K,
                            trajectory_generator,
                            segment_start,
                            previous_q_cmd);
  }
  return RunCommandedPoseIk(
      row, previous_row, model, config, F_T_EE, EE_T_K, previous_q_cmd);
}

void UpdateMetrics(const std::array<double, 7>& estimate,
                   const std::array<double, 7>& truth,
                   const Pose& estimated_pose,
                   const Pose& target_pose,
                   MetricsAccumulator* metrics) {
  ++metrics->count;
  for (size_t i = 0; i < 7; ++i) {
    const double abs_error = std::abs(estimate[i] - truth[i]);
    metrics->mae_sum += abs_error;
    metrics->mae_joint_sum[i] += abs_error;
    metrics->max_abs_joint[i] = std::max(metrics->max_abs_joint[i], abs_error);
    metrics->max_abs_error = std::max(metrics->max_abs_error, abs_error);
  }
  const double pos_error = PositionErrorNorm(estimated_pose, target_pose);
  const double rot_error = RotationErrorNorm(estimated_pose, target_pose);
  metrics->pose_position_error_sum += pos_error;
  metrics->pose_position_error_max = std::max(metrics->pose_position_error_max, pos_error);
  metrics->pose_rotation_error_sum += rot_error;
  metrics->pose_rotation_error_max = std::max(metrics->pose_rotation_error_max, rot_error);
}

void PrintMetrics(const MetricsAccumulator& metrics) {
  if (metrics.count == 0) {
    std::cout << "No ground-truth q_cmd rows were available for comparison.\n";
    return;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Compared rows: " << metrics.count << "\n";
  std::cout << "Joint MAE overall: " << (metrics.mae_sum / (static_cast<double>(metrics.count) * 7.0))
            << " rad\n";
  std::cout << "Joint max abs error overall: " << metrics.max_abs_error << " rad\n";
  for (size_t i = 0; i < 7; ++i) {
    std::cout << "  joint_" << i
              << " mae=" << (metrics.mae_joint_sum[i] / static_cast<double>(metrics.count))
              << " max=" << metrics.max_abs_joint[i] << "\n";
  }
  std::cout << "Estimated pose error vs commanded_target_state: mean_pos="
            << (metrics.pose_position_error_sum / static_cast<double>(metrics.count))
            << " m max_pos=" << metrics.pose_position_error_max
            << " m mean_rot=" << (metrics.pose_rotation_error_sum / static_cast<double>(metrics.count))
            << " rad max_rot=" << metrics.pose_rotation_error_max << " rad\n";
}

std::string JoinVector(const std::array<double, 7>& values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << std::fixed << std::setprecision(9) << values[i];
  }
  out << ']';
  return out.str();
}

}  // namespace

int OfflineQcmdValidatorMain(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  AppConfig app_config;
  std::string config_error;
  if (!LoadAppConfig(options.config_dir, &app_config, &config_error)) {
    std::cerr << "Failed to load config: " << config_error << "\n";
    return 1;
  }
  if (!options.robot_ip_override.empty()) {
    app_config.bridge.robot_ip = options.robot_ip_override;
  }

  std::cout << "Connecting read-only to robot at " << app_config.bridge.robot_ip << "...\n";
  franka::Robot robot(app_config.bridge.robot_ip);
  franka::Model model = robot.loadModel();
  const franka::RobotState state = robot.readOnce();
  const std::array<double, 16> F_T_EE = state.F_T_EE;
  const std::array<double, 16> EE_T_K = state.EE_T_K;
  std::cout << "Loaded libfranka model and current tool transforms.\n";

  std::ifstream input(options.input_path);
  if (!input) {
    std::cerr << "Failed to open input: " << options.input_path << "\n";
    return 1;
  }

  std::ofstream output;
  if (options.write_output) {
    const std::filesystem::path input_path = std::filesystem::weakly_canonical(options.input_path);
    const std::filesystem::path output_path = options.output_path;
    std::error_code ec;
    const std::filesystem::path output_parent = output_path.parent_path();
    if (!output_parent.empty()) {
      std::filesystem::create_directories(output_parent, ec);
      if (ec) {
        std::cerr << "Failed to create output directory " << output_parent
                  << ": " << ec.message() << "\n";
        return 1;
      }
    }
    if (std::filesystem::exists(output_path)) {
      const std::filesystem::path output_canonical = std::filesystem::weakly_canonical(output_path);
      if (input_path == output_canonical) {
        std::cerr << "Refusing to overwrite input file with output: " << output_canonical << "\n";
        return 1;
      }
    } else if (input_path == output_path.lexically_normal()) {
      std::cerr << "Refusing to overwrite input file with output: " << output_path << "\n";
      return 1;
    }
    output.open(options.output_path);
    if (!output) {
      std::cerr << "Failed to open output: " << options.output_path << "\n";
      return 1;
    }
  }

  JointPositionTrajectoryGenerator trajectory_generator(app_config.bridge);
  std::optional<ObservationRow> previous_row;
  std::array<double, 7> previous_q_cmd{};
  MetricsAccumulator metrics;
  std::string line;
  size_t line_no = 0;
  size_t processed_rows = 0;
  size_t skipped_rows = 0;
  size_t printed_parse_errors = 0;
  size_t parsed_rows = 0;
  size_t segment_starts = 0;

  while (std::getline(input, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }
    nlohmann::json root;
    try {
      root = nlohmann::json::parse(line);
    } catch (const std::exception& e) {
      std::cerr << options.input_path << ":" << line_no << ": invalid JSON: " << e.what() << "\n";
      return 1;
    }

    ObservationRow row;
    std::string parse_error;
    if (!ParseObservationRow(root, &row, &parse_error)) {
      ++skipped_rows;
      if (printed_parse_errors < 5) {
        std::cerr << options.input_path << ":" << line_no << ": parse skipped: " << parse_error << "\n";
        ++printed_parse_errors;
      }
      continue;
    }
    ++parsed_rows;
    if (parsed_rows <= options.skip_rows) {
      previous_row = row;
      previous_q_cmd = row.q;
      continue;
    }

    const bool segment_start = IsSegmentStart(row, previous_row ? &(*previous_row) : nullptr);
    if (segment_start) {
      ++segment_starts;
      trajectory_generator.Reset();
      previous_q_cmd = ComputeSegmentSeed(row, model, app_config.bridge, F_T_EE, EE_T_K);
    }

    const std::array<double, 7> estimate = ComputeEstimate(row,
                                                            previous_row ? &(*previous_row) : nullptr,
                                                            model,
                                                            app_config.bridge,
                                                            F_T_EE,
                                                            EE_T_K,
                                                            options.method,
                                                            &trajectory_generator,
                                                            segment_start,
                                                            &previous_q_cmd);
    const Pose estimated_pose = ForwardPose(model, estimate, F_T_EE, EE_T_K);
    if (row.q_cmd_truth.has_value()) {
      UpdateMetrics(estimate, *row.q_cmd_truth, estimated_pose, row.commanded_target_pose, &metrics);
    }

    if (options.write_output) {
      root["backfilled_q_cmd"] = estimate;
      root["backfilled_q_cmd_method"] = options.method;
      output << root.dump() << '\n';
    }

    previous_row = row;
    ++processed_rows;
    if (options.max_rows > 0 && processed_rows >= options.max_rows) {
      break;
    }
  }

  std::cout << "Processed rows: " << processed_rows << " skipped rows: " << skipped_rows
            << " segment_starts: " << segment_starts << "\n";
  PrintMetrics(metrics);
  if (processed_rows > 0) {
    std::cout << "Last estimate: " << JoinVector(previous_q_cmd) << "\n";
  }
  return 0;
}

}  // namespace teleop

int main(int argc, char** argv) {
  try {
    return teleop::OfflineQcmdValidatorMain(argc, argv);
  } catch (const franka::Exception& e) {
    std::cerr << "libfranka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
