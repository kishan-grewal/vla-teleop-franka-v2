#include "franka_controller.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include "gripper.h"
#include "math_utils.h"
#include "safety.h"
#include "teleop_mapper.h"
#include "teleop_state_machine.h"
#include "trace_logger.h"

namespace teleop {
namespace {

constexpr double kRehomeLiftClearanceAboveWorkspaceMinM = 0.28;
constexpr double kRehomeTopWorkspaceMarginM = 0.08;
constexpr double kRehomeLiftTriggerToleranceM = 0.008;
constexpr double kRehomeLiftSpeedMps = 0.04;
constexpr double kRehomeJointSpeedRadPerS = 0.35;
constexpr double kRehomeStopVelocityToleranceRadPerS = 0.02;
constexpr double kRehomeStopCommandToleranceRad = 1e-3;
constexpr uint32_t kRehomeStopSettledCycles = 60;
constexpr double kStartupHomeArrivalToleranceRad = 5e-3;
constexpr uint64_t kPlannerTargetGraceNs = 35000000ull;  // 35 ms
constexpr uint64_t kEpisodeMarkerNs = 250000000ull;  // 250 ms
constexpr uint64_t kPostRehomeMotionInhibitNs = 400000000ull;  // 400 ms settle window
constexpr std::array<double, 7> kPandaJointLowerLimitsRad{
    {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
constexpr std::array<double, 7> kPandaJointUpperLimitsRad{
    {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
constexpr double kJointLimitMarginRad = 0.02;
constexpr uint32_t kCartesianContactDebounceCycles = 15;  // ~15 ms at 1 kHz

template <size_t N>
bool AnyPositive(const std::array<double, N>& values) {
  for (double value : values) {
    if (value > 0.0) {
      return true;
    }
  }
  return false;
}

double QuinticSmoothstep(double tau) {
  const double clamped_tau = std::clamp(tau, 0.0, 1.0);
  const double tau2 = clamped_tau * clamped_tau;
  const double tau3 = tau2 * clamped_tau;
  return tau3 * (10.0 + clamped_tau * (-15.0 + 6.0 * clamped_tau));
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

Pose ApplyPolicyActionDelta(const Pose& current_pose, const TeleopAction& action) {
  Pose desired = current_pose;
  for (size_t i = 0; i < 3; ++i) {
    desired.p[i] += action.delta_translation_m[i];
  }

  const Eigen::Vector3d rotation_delta = ToEigen(action.delta_rotation_rad);
  const double rotation_norm = rotation_delta.norm();
  if (rotation_norm > 1e-12 && std::isfinite(rotation_norm)) {
    const Eigen::Vector3d axis = rotation_delta / rotation_norm;
    const Eigen::Quaterniond q_step(Eigen::AngleAxisd(rotation_norm, axis));
    desired.q = ToArrayQuat(q_step * ToEigenQuat(current_pose.q));
  }
  return desired;
}

Pose PoseFromJointPositions(const franka::Model& model,
                            const RobotSnapshot& snapshot,
                            const std::array<double, 7>& q) {
  return MatrixToPose(model.pose(franka::Frame::kEndEffector, q, snapshot.F_T_EE, snapshot.EE_T_K));
}

double MaxAbsJointDelta(const std::array<double, 7>& a, const std::array<double, 7>& b) {
  double max_abs = 0.0;
  for (size_t i = 0; i < 7; ++i) {
    max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
  }
  return max_abs;
}

double JointDeltaNorm(const std::array<double, 7>& a, const std::array<double, 7>& b) {
  double sum_sq = 0.0;
  for (size_t i = 0; i < 7; ++i) {
    const double delta = a[i] - b[i];
    sum_sq += delta * delta;
  }
  return std::sqrt(sum_sq);
}

bool PoseInsideWorkspace(const Pose& pose, const SafetyLimits& limits) {
  for (size_t i = 0; i < 3; ++i) {
    if (pose.p[i] < limits.workspace_min[i] || pose.p[i] > limits.workspace_max[i]) {
      return false;
    }
  }
  return true;
}

bool ValidatePolicyJointTarget(const franka::Model& model,
                               const RobotSnapshot& snapshot,
                               const std::array<double, 7>& candidate_q,
                               const TeleopBridgeConfig& config,
                               std::array<double, 7>* validated_q,
                               Pose* target_pose,
                               FaultFlags* faults) {
  *validated_q = ClampToJointLimits(candidate_q);
  const double max_abs_joint_delta = MaxAbsJointDelta(*validated_q, snapshot.q_d);
  const double joint_delta_norm = JointDeltaNorm(*validated_q, snapshot.q_d);
  if (max_abs_joint_delta > config.policy.max_joint_delta_rad ||
      joint_delta_norm > config.policy.max_joint_distance_rad) {
    faults->jump_rejected = true;
    return false;
  }

  *target_pose = PoseFromJointPositions(model, snapshot, *validated_q);
  if (!PoseInsideWorkspace(*target_pose, config.safety)) {
    faults->workspace_clamped = true;
    return false;
  }

  const Eigen::Vector3d tcp_translation_jump =
      ToEigen(target_pose->p) - ToEigen(snapshot.tcp_pose_d.p);
  const Eigen::Vector3d tcp_rotation_jump =
      QuaternionErrorAngleAxis(ToEigenQuat(snapshot.tcp_pose_d.q), ToEigenQuat(target_pose->q));
  if (tcp_translation_jump.norm() > config.safety.jump_reject_translation_m ||
      tcp_rotation_jump.norm() > config.safety.jump_reject_rotation_rad) {
    faults->jump_rejected = true;
    return false;
  }

  return true;
}

bool SolveIkStep(const franka::Model& model,
                 const RobotSnapshot& snapshot,
                 const Pose& desired_pose,
                 const TeleopBridgeConfig& config,
                 ControlMode control_mode,
                 std::array<double, 7>* target_q,
                 double* manipulability);

void ConfigureConservativeBehavior(franka::Robot* robot) {
  robot->setCollisionBehavior(
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
  // Keep the internal joint impedance less aggressive to reduce audible chatter
  // when teleop commands dither around a steady target.
  robot->setJointImpedance({{1500.0, 1500.0, 1500.0, 1250.0, 1250.0, 1000.0, 1000.0}});
}

RobotSnapshot ToSnapshot(const franka::RobotState& state) {
  RobotSnapshot snapshot{};
  snapshot.timestamp_ns = MonotonicNowNs();
  snapshot.q = state.q;
  snapshot.q_d = state.q_d;
  snapshot.dq = state.dq;
  snapshot.tcp_pose = MatrixToPose(state.O_T_EE);
  snapshot.tcp_pose_d = MatrixToPose(state.O_T_EE_d);
  snapshot.F_T_EE = state.F_T_EE;
  snapshot.EE_T_K = state.EE_T_K;
  snapshot.in_reflex = (state.robot_mode == franka::RobotMode::kReflex);
  snapshot.robot_ok = !snapshot.in_reflex;
  return snapshot;
}

bool RecoverRobotIfNeeded(franka::Robot* robot) {
  const franka::RobotState state = robot->readOnce();
  if (state.robot_mode != franka::RobotMode::kReflex) {
    return true;
  }
  robot->automaticErrorRecovery();
  return true;
}

bool CommandGripperOpen(franka::Gripper* gripper,
                        const GripperConfig& config,
                        std::atomic<double>* measured_gripper_width_m) {
  if (gripper == nullptr) {
    return true;
  }

  try {
    const bool move_ok = gripper->move(config.max_width_m, config.speed_mps);
    const franka::GripperState state = gripper->readOnce();
    if (measured_gripper_width_m != nullptr) {
      measured_gripper_width_m->store(state.width, std::memory_order_release);
    }
    if (!move_ok) {
      std::cerr << "Gripper open command returned false during homing.\n";
    }
    return move_ok;
  } catch (const std::exception& e) {
    std::cerr << "Gripper open command failed during homing: " << e.what() << "\n";
    return false;
  }
}

bool MoveToLiftPose(franka::Robot* robot,
                    const SafetyLimits& safety,
                    std::atomic<bool>* stop_requested) {
  const franka::RobotState state0 = robot->readOnce();
  const double current_z = state0.O_T_EE[14];
  const double workspace_min_z = safety.workspace_min[2];
  double lift_target_z = std::max(current_z, workspace_min_z + kRehomeLiftClearanceAboveWorkspaceMinM);
  if (safety.enforce_workspace_limits_during_rehome) {
    const double workspace_max_z = safety.workspace_max[2];
    lift_target_z = std::clamp(
        lift_target_z, workspace_min_z, workspace_max_z - kRehomeTopWorkspaceMarginM);
  }
  if (lift_target_z <= current_z + kRehomeLiftTriggerToleranceM) {
    return true;
  }

  const std::array<double, 16> start_pose = state0.O_T_EE_d;
  const std::array<double, 2> elbow = state0.elbow_d;
  const double duration_s =
      std::max(2.0, 1.875 * std::abs(lift_target_z - current_z) / kRehomeLiftSpeedMps);
  double elapsed_s = 0.0;

  robot->control(
      [&](const franka::RobotState&, franka::Duration period) -> franka::CartesianPose {
        elapsed_s += period.toSec();
        const double tau = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
        const double smooth = QuinticSmoothstep(tau);
        std::array<double, 16> commanded_pose = start_pose;
        commanded_pose[14] = start_pose[14] + smooth * (lift_target_z - start_pose[14]);
        franka::CartesianPose out(commanded_pose, elbow);
        if (tau >= 1.0 || stop_requested->load(std::memory_order_acquire)) {
          return franka::MotionFinished(out);
        }
        return out;
      },
      franka::ControllerMode::kJointImpedance,
      true,
      100.0);
  return !stop_requested->load(std::memory_order_acquire);
}

bool MoveToHomePose(franka::Robot* robot,
                    const std::array<double, 7>& q_goal,
                    std::atomic<bool>* stop_requested) {
  const std::array<double, 7> q_goal_clamped = ClampToJointLimits(q_goal);
  const franka::RobotState state0 = robot->readOnce();
  const std::array<double, 7> q_start = state0.q_d;
  double max_error = 0.0;
  for (size_t i = 0; i < 7; ++i) {
    max_error = std::max(max_error, std::abs(q_goal_clamped[i] - q_start[i]));
  }
  if (max_error < kStartupHomeArrivalToleranceRad) {
    return true;
  }

  const double duration_s =
      std::max(2.5, 1.875 * max_error / kRehomeJointSpeedRadPerS);
  double elapsed_s = 0.0;
  robot->control(
      [&](const franka::RobotState&, franka::Duration period) -> franka::JointPositions {
        elapsed_s += period.toSec();
        const double tau = std::clamp(elapsed_s / duration_s, 0.0, 1.0);
        const double smooth = QuinticSmoothstep(tau);
        std::array<double, 7> q_cmd{};
        for (size_t i = 0; i < 7; ++i) {
          q_cmd[i] = q_start[i] + smooth * (q_goal_clamped[i] - q_start[i]);
        }
        franka::JointPositions out(q_cmd);
        if (tau >= 1.0 || stop_requested->load(std::memory_order_acquire)) {
          return franka::MotionFinished(out);
        }
        return out;
      },
      franka::ControllerMode::kJointImpedance,
      true,
      100.0);
  return !stop_requested->load(std::memory_order_acquire);
}

bool MoveToSafeHomeRoute(franka::Robot* robot,
                         const TeleopBridgeConfig& config,
                         std::atomic<bool>* stop_requested) {
  if (!MoveToLiftPose(robot, config.safety, stop_requested)) {
    return false;
  }
  if (!MoveToHomePose(robot, config.ik.nullspace_joint_positions_rad, stop_requested)) {
    return false;
  }
  if (!MoveToHomePose(robot, config.teleop.start_joint_positions_rad, stop_requested)) {
    return false;
  }
  return true;
}

void PublishHoldTarget(const RobotSnapshot& snapshot,
                       double target_gripper_width_m,
                       TeleopState teleop_state,
                       LatestPlannedTargetBuffer* planned_target_buffer) {
  PlannedTarget hold_target{};
  hold_target.target_timestamp_ns = MonotonicNowNs();
  hold_target.target_q = snapshot.q;
  hold_target.desired_tcp_pose = snapshot.tcp_pose;
  hold_target.target_gripper_width_m = target_gripper_width_m;
  hold_target.control_mode = ControlMode::kHold;
  hold_target.teleop_active = false;
  hold_target.target_fresh = false;
  hold_target.teleop_state = teleop_state;
  planned_target_buffer->Publish(hold_target);
}

double MapTriggerToWidth(const GripperConfig& config, double trigger) {
  const double clamped = Clamp01(trigger);
  const double span = config.max_width_m - config.min_width_m;
  return config.max_width_m - clamped * span;
}

double ClampWidth(const GripperConfig& config, double width) {
  return std::clamp(width, config.min_width_m, config.max_width_m);
}

double MapStateToWidth(const GripperConfig& config, GripperState state) {
  return state == GripperState::kClose ? config.min_width_m : config.max_width_m;
}

bool GripperStateSatisfiesDesired(GripperState actual, GripperState desired) {
  if (desired == GripperState::kOpen) {
    return actual == GripperState::kOpen;
  }
  if (desired == GripperState::kClose) {
    return actual == GripperState::kClose || actual == GripperState::kHold;
  }
  return actual == desired;
}

GripperState WidthToAnalogState(const GripperConfig& config, double width, bool closing_stalled) {
  if (closing_stalled) {
    return GripperState::kHold;
  }
  if (width >= (config.max_width_m - config.width_tolerance_m)) {
    return GripperState::kOpen;
  }
  if (width <= (config.min_width_m + config.width_tolerance_m)) {
    return GripperState::kClose;
  }
  return GripperState::kHold;
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

  // Build the next reference from the commanded state that libfranka is already
  // tracking, instead of re-seeding from measured joint motion every planner tick.
  std::array<double, 7> q_next = snapshot.q_d;
  Pose tcp_pose_next = snapshot.tcp_pose_d;
  const Eigen::Matrix<double, 7, 1> q_nullspace_target = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(
      config.ik.nullspace_joint_positions_rad.data());
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
                               bool apply_motion,
                               std::array<double, 7>* target_delta,
                               std::array<double, 7>* filtered_delta,
                               std::array<double, 7>* command_delta,
                               std::array<uint8_t, 7>* clamp_saturated) {
    if (!initialized_) {
      q_ref_ = reference_q;
      dq_ref_.fill(0.0);
      ddq_ref_.fill(0.0);
      hold_target_q_ = q_ref_;
      initialized_ = true;
    }

    if (!apply_motion) {
      if (motion_active_) {
        // Latch the robot's current commanded reference to avoid a small rewind
        // when teleop gating drops between frames.
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
      // Coast-down mode: damp residual trajectory velocity to zero without
      // pulling back toward a latched position target.
      std::array<double, 7> q_out = q_ref_;
      std::array<double, 7> dq_out = dq_ref_;
      std::array<double, 7> ddq_out = ddq_ref_;
      target_delta->fill(0.0);
      command_delta->fill(0.0);
      clamp_saturated->fill(0);

      for (size_t i = 0; i < 7; ++i) {
        (*filtered_delta)[i] = q_ref_[i] - reference_q[i];
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
        (*command_delta)[i] = q_out[i] - reference_q[i];
        if (std::abs(velocity_unclamped - dq_out[i]) > 1e-12 || std::abs(step_unclamped - step) > 1e-12) {
          (*clamp_saturated)[i] = 1;
        }
      }

      q_ref_ = q_out;
      dq_ref_ = dq_out;
      ddq_ref_ = ddq_out;
      hold_target_q_ = q_ref_;
      return q_out;
    }

    const std::array<double, 7>& active_target_q = target_q;

    for (size_t i = 0; i < 7; ++i) {
      (*target_delta)[i] = active_target_q[i] - reference_q[i];
      (*filtered_delta)[i] = q_ref_[i] - reference_q[i];
    }

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
      command_delta->fill(0.0);
      clamp_saturated->fill(0);
      return q_ref_;
    }

    if (within_hold_band) {
      hold_active_ = true;
      dq_ref_.fill(0.0);
      ddq_ref_.fill(0.0);
      command_delta->fill(0.0);
      clamp_saturated->fill(0);
      return q_ref_;
    }

    hold_active_ = false;
    std::array<double, 7> q_out = q_ref_;
    std::array<double, 7> dq_out = dq_ref_;
    std::array<double, 7> ddq_out = ddq_ref_;
    command_delta->fill(0.0);
    clamp_saturated->fill(0);

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

      (*command_delta)[i] = q_out[i] - reference_q[i];
      if (std::abs(velocity_unclamped - dq_out[i]) > 1e-12 || std::abs(step_unclamped - step) > 1e-12) {
        (*clamp_saturated)[i] = 1;
      }
    }

    q_ref_ = q_out;
    dq_ref_ = dq_out;
    ddq_ref_ = ddq_out;
    for (size_t i = 0; i < 7; ++i) {
      (*filtered_delta)[i] = q_ref_[i] - reference_q[i];
    }
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

void PlannerLoop(const TeleopBridgeConfig& config,
                 const franka::Model& model,
                 const LatestCommandBuffer* command_buffer,
                 const LatestPolicyActionBuffer* policy_action_buffer,
                 const LatestRobotStateBuffer* robot_state_buffer,
                 LatestPlannedTargetBuffer* planned_target_buffer,
                 TraceRecorder* trace_recorder,
                 uint32_t trace_decimation,
                 const std::atomic<GripperState>* active_gripper_state,
                 std::atomic<GripperState>* desired_gripper_state,
                 std::atomic<double>* desired_gripper_width_m,
                 std::atomic<uint64_t>* requested_rehome_request_id,
                 const std::atomic<uint64_t>* completed_rehome_request_id,
                 const std::atomic<bool>* rehome_in_progress,
                 std::atomic<bool>* stop_requested) {
  try {
    const auto sleep_period =
        std::chrono::microseconds(static_cast<int64_t>(1e6 / config.teleop.planner_rate_hz));

    SafetyFilter safety(config.safety, config.teleop.planner_rate_hz);
    GripperController gripper_controller;
    const GripperState initial_gripper_state =
        active_gripper_state->load(std::memory_order_acquire);
    gripper_controller.Reset(initial_gripper_state);
    TeleopMapper mapper(config);
    TeleopStateMachine state_machine;
    bool deadman_latched = false;
    const bool use_a_button_for_episode_markers = !config.teleop.a_button_toggles_robot_control;
    bool last_button_a = false;
    bool last_button_b = false;
    uint64_t episode_start_marker_until_ns = 0;
    uint64_t episode_end_marker_until_ns = 0;
    bool has_recent_target = false;
    uint64_t last_valid_target_ns = 0;
    std::array<double, 7> last_valid_target_q{};
    Pose last_valid_desired_pose{};
    ControlMode last_valid_control_mode = ControlMode::kHold;
    double last_valid_manipulability = 0.0;
    uint64_t last_seen_completed_rehome_request_id =
        completed_rehome_request_id->load(std::memory_order_acquire);
    uint64_t planner_last_ns = 0;
    uint64_t planner_trace_counter = 0;
    const uint32_t planner_trace_decimation = std::max<uint32_t>(1, trace_decimation);

    while (!stop_requested->load(std::memory_order_acquire)) {
      const uint64_t now_ns = MonotonicNowNs();
      const uint64_t loop_dt_ns = (planner_last_ns == 0) ? 0 : (now_ns - planner_last_ns);
      planner_last_ns = now_ns;
      const bool policy_control = config.control_source == ControlSource::kPolicy;
      const XRCommand xr_cmd = command_buffer->ReadLatest();
      const PolicyActionCommand policy_cmd =
          policy_action_buffer != nullptr ? policy_action_buffer->ReadLatest() : PolicyActionCommand{};
      const RobotSnapshot robot = robot_state_buffer->ReadLatest();
      const uint64_t input_timestamp_ns = policy_control ? policy_cmd.timestamp_ns : xr_cmd.timestamp_ns;
      const uint64_t input_sequence_id = policy_control ? policy_cmd.sequence_id : xr_cmd.sequence_id;

      PlannedTarget planned{};
      planned.target_timestamp_ns = now_ns;
      planned.target_q = robot.q;
      planned.desired_tcp_pose = robot.tcp_pose;
      planned.control_mode = ControlMode::kHold;
      const double policy_gripper_command = Clamp01(policy_cmd.action.gripper_command);
      const bool policy_rehome_requested = policy_control && policy_cmd.request_rehome;
      const uint64_t completed_rehome_id_snapshot =
          completed_rehome_request_id->load(std::memory_order_acquire);
      if (completed_rehome_id_snapshot != last_seen_completed_rehome_request_id) {
        gripper_controller.Reset(GripperState::kOpen);
        // Re-home moves the robot outside the normal teleop path, so any cached
        // IK/planner target from before the move is stale and must not be reused
        // as the next XR anchor.
        has_recent_target = false;
        last_valid_target_ns = 0;
        last_valid_target_q = robot.q;
        last_valid_desired_pose = robot.tcp_pose;
        last_valid_control_mode = ControlMode::kHold;
        last_valid_manipulability = 0.0;
        last_seen_completed_rehome_request_id = completed_rehome_id_snapshot;
      }
      const double gripper_trigger =
          policy_control ? policy_gripper_command : Clamp01(xr_cmd.gripper_trigger_value);
      GripperState desired_state = GripperState::kOpen;
      double gripper_command = 0.0;
      double desired_gripper_width = config.gripper.max_width_m;
      if (policy_control) {
        gripper_command = policy_gripper_command;
        desired_state = policy_gripper_command >= 0.5 ? GripperState::kClose : GripperState::kOpen;
        desired_gripper_width = MapStateToWidth(config.gripper, desired_state);
      } else if (config.gripper.command_mode == GripperCommandMode::kAnalog) {
        desired_gripper_width = MapTriggerToWidth(config.gripper, gripper_trigger);
        gripper_command = gripper_trigger;
        const double midpoint = 0.5 * (config.gripper.min_width_m + config.gripper.max_width_m);
        desired_state = desired_gripper_width <= midpoint ? GripperState::kClose : GripperState::kOpen;
      } else {
        desired_state = gripper_controller.UpdateDesiredState(config.gripper, gripper_trigger, now_ns);
        gripper_command = desired_state == GripperState::kClose ? 1.0 : 0.0;
        desired_gripper_width = MapStateToWidth(config.gripper, desired_state);
      }
      const double control_value =
          policy_control ? ((policy_cmd.enabled && !policy_rehome_requested) ? 1.0 : 0.0)
                         : Clamp01(xr_cmd.control_trigger_value);

      bool has_target = false;
      bool safe_target = false;
      bool ik_ok = false;
      Pose desired_pose = robot.tcp_pose;
      Pose safe_pose = robot.tcp_pose;
      std::array<double, 7> q_target = robot.q;

      auto publish_trace = [&]() {
        if (trace_recorder == nullptr) {
          return;
        }
        if ((planner_trace_counter++ % planner_trace_decimation) != 0) {
          return;
        }

        PlannerTraceSample trace{};
        trace.timestamp_ns = now_ns;
        trace.loop_dt_ns = loop_dt_ns;
        trace.xr_timestamp_ns = input_timestamp_ns;
        trace.xr_sequence_id = input_sequence_id;
        trace.packet_age_ns = planned.packet_age_ns;
        trace.teleop_state = static_cast<int>(planned.teleop_state);
        trace.control_mode = static_cast<int>(planned.control_mode);
        trace.teleop_active = planned.teleop_active;
        trace.target_fresh = planned.target_fresh;
        trace.deadman_latched = deadman_latched;
        trace.has_target = has_target;
        trace.safe_target = safe_target;
        trace.ik_ok = ik_ok;
        trace.faults = planned.faults;
        trace.control_trigger_value = control_value;
        trace.xr_position = policy_control ? robot.tcp_pose.p : xr_cmd.right_controller_pose.p;
        trace.desired_position = desired_pose.p;
        trace.safe_position = safe_pose.p;
        trace.robot_position = robot.tcp_pose.p;
        trace.requested_delta_translation = planned.requested_action.delta_translation_m;
        trace.requested_delta_rotation = planned.requested_action.delta_rotation_rad;
        trace.safe_delta_translation = {
            safe_pose.p[0] - robot.tcp_pose.p[0],
            safe_pose.p[1] - robot.tcp_pose.p[1],
            safe_pose.p[2] - robot.tcp_pose.p[2]};
        trace.safe_delta_rotation =
            ToArray3(QuaternionErrorAngleAxis(ToEigenQuat(robot.tcp_pose.q), ToEigenQuat(safe_pose.q)));
        trace.q_robot = robot.q;
        trace.q_raw_target = q_target;
        trace.q_planned = planned.target_q;
        trace.manipulability = planned.manipulability;
        trace_recorder->PushPlanner(trace);
      };

      if (robot.timestamp_ns == 0) {
        planned_target_buffer->Publish(planned);
        publish_trace();
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      const uint64_t packet_age_ns =
          input_timestamp_ns != 0 && now_ns > input_timestamp_ns ? (now_ns - input_timestamp_ns)
                                                                 : std::numeric_limits<uint64_t>::max();
      planned.packet_age_ns = packet_age_ns;
      const double packet_age_s = static_cast<double>(packet_age_ns) * 1e-9;

      StateInputs inputs{};
      inputs.xr_stream_healthy =
          input_timestamp_ns != 0 &&
          (policy_control ? packet_age_s <= config.policy.command_timeout_s
                          : safety.IsStreamHealthy(packet_age_s));
      if (!inputs.xr_stream_healthy) {
        deadman_latched = false;
      } else if (policy_control) {
        deadman_latched = policy_cmd.enabled && !policy_rehome_requested;
      } else if (deadman_latched) {
        if (control_value <= config.teleop.control_trigger_release_threshold) {
          deadman_latched = false;
        }
      } else if (control_value >= config.teleop.control_trigger_threshold) {
        deadman_latched = true;
      }
      const bool button_a = inputs.xr_stream_healthy ? xr_cmd.button_a : false;
      const bool button_b = inputs.xr_stream_healthy ? xr_cmd.button_b : false;
      if (policy_control) {
        if (inputs.xr_stream_healthy && policy_cmd.episode_start) {
          episode_start_marker_until_ns = now_ns + kEpisodeMarkerNs;
        }
        if (inputs.xr_stream_healthy && policy_cmd.episode_end) {
          episode_end_marker_until_ns = now_ns + kEpisodeMarkerNs;
        }
      } else if (use_a_button_for_episode_markers) {
        if (button_a && !last_button_a) {
          episode_start_marker_until_ns = now_ns + kEpisodeMarkerNs;
        }
        if (button_b && !last_button_b) {
          episode_end_marker_until_ns = now_ns + kEpisodeMarkerNs;
          const uint64_t requested_id =
              requested_rehome_request_id->load(std::memory_order_acquire);
          const uint64_t completed_id =
              completed_rehome_request_id->load(std::memory_order_acquire);
          if (requested_id <= completed_id) {
            requested_rehome_request_id->store(completed_id + 1, std::memory_order_release);
          }
        }
      } else if (button_b && !last_button_b) {
        episode_end_marker_until_ns = now_ns + kEpisodeMarkerNs;
        const uint64_t requested_id =
            requested_rehome_request_id->load(std::memory_order_acquire);
        const uint64_t completed_id =
            completed_rehome_request_id->load(std::memory_order_acquire);
        if (requested_id <= completed_id) {
          requested_rehome_request_id->store(completed_id + 1, std::memory_order_release);
        }
      }
      planned.episode_start =
          episode_start_marker_until_ns != 0 && now_ns <= episode_start_marker_until_ns;
      planned.episode_end =
          episode_end_marker_until_ns != 0 && now_ns <= episode_end_marker_until_ns;
      last_button_a = button_a;
      last_button_b = button_b;
      const uint64_t requested_rehome_id =
          requested_rehome_request_id->load(std::memory_order_acquire);
      const uint64_t completed_rehome_id = completed_rehome_id_snapshot;
      const bool rehome_requested =
          requested_rehome_id > completed_rehome_id ||
          rehome_in_progress->load(std::memory_order_acquire);
      const GripperState latched_gripper_state =
          desired_gripper_state->load(std::memory_order_acquire);
      const double latched_gripper_width =
          desired_gripper_width_m->load(std::memory_order_acquire);
      if (rehome_requested) {
        deadman_latched = false;
        desired_state = latched_gripper_state;
        desired_gripper_width = latched_gripper_width;
      }
      planned.target_gripper_width_m = desired_gripper_width;
      planned.requested_action.gripper_command = gripper_command;
      if (!rehome_requested) {
        desired_gripper_state->store(desired_state, std::memory_order_release);
        desired_gripper_width_m->store(desired_gripper_width, std::memory_order_release);
      }
      inputs.deadman_pressed = rehome_requested ? false : deadman_latched;
      inputs.robot_ok = robot.robot_ok;
      inputs.fault_requested = false;
      inputs.clear_fault_requested = true;
      planned.teleop_state = state_machine.Update(inputs);
      planned.teleop_active = (planned.teleop_state == TeleopState::kTeleopActive);

      planned.faults.robot_not_ready = !inputs.robot_ok;
      if (!inputs.xr_stream_healthy) {
        planned.faults.packet_timeout = true;
      }

      if (!planned.teleop_active || rehome_requested) {
        if (rehome_requested) {
          planned.control_mode = ControlMode::kHold;
          planned.teleop_active = false;
          planned.target_fresh = false;
          planned.desired_tcp_pose = robot.tcp_pose;
          planned.requested_action = TeleopAction{};
          planned.requested_action.gripper_command = gripper_command;
        } else if (last_valid_target_ns != 0) {
          planned.desired_tcp_pose = last_valid_desired_pose;
        }
        has_recent_target = false;
        if (rehome_requested) {
          mapper.Reset();
        } else if (!policy_control && inputs.xr_stream_healthy) {
          mapper.Reanchor(planned.desired_tcp_pose, xr_cmd);
        } else {
          mapper.Reset();
        }
        planned_target_buffer->Publish(planned);
        publish_trace();
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      planned.control_mode = config.teleop.control_mode;
      TeleopAction requested_action{};
      if (policy_control) {
        has_target = inputs.xr_stream_healthy && policy_cmd.enabled && !policy_rehome_requested;
        requested_action = policy_cmd.action;
        requested_action.gripper_command = gripper_command;
        if (requested_action.action_space == ActionSpace::kJointPositionAbsolute) {
          desired_pose = robot.tcp_pose;
        } else {
          desired_pose = ApplyPolicyActionDelta(robot.tcp_pose, requested_action);
        }
      } else {
        has_target = mapper.ComputeTargetPose(robot.tcp_pose,
                                              xr_cmd,
                                              true,
                                              planned.control_mode,
                                              &desired_pose,
                                              &requested_action);
      }
      planned.requested_action = requested_action;
      planned.requested_action.gripper_command = gripper_command;
      if (!has_target) {
        const bool can_reuse_target =
            has_recent_target && now_ns > last_valid_target_ns &&
            (now_ns - last_valid_target_ns) <= kPlannerTargetGraceNs;
        if (can_reuse_target) {
          planned.control_mode = last_valid_control_mode;
          planned.target_q = last_valid_target_q;
          planned.desired_tcp_pose = last_valid_desired_pose;
          planned.manipulability = last_valid_manipulability;
          planned.target_fresh = true;
        } else {
          planned.control_mode = ControlMode::kHold;
        }
        planned_target_buffer->Publish(planned);
        publish_trace();
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      const bool use_direct_joint_target =
          policy_control && requested_action.action_space == ActionSpace::kJointPositionAbsolute;
      if (use_direct_joint_target) {
        safe_target = ValidatePolicyJointTarget(
            model, robot, requested_action.joint_positions_rad, config, &q_target, &safe_pose, &planned.faults);
        desired_pose = safe_pose;
      } else if (policy_control) {
        safe_target = safety.FilterTargetPose(robot.tcp_pose, desired_pose, 0.0, &planned.faults, &safe_pose);
      } else {
        safe_target = safety.FilterTargetPose(robot.tcp_pose,
                                              desired_pose,
                                              packet_age_s,
                                              &planned.faults,
                                              &safe_pose);
      }
      planned.desired_tcp_pose = safe_pose;
      if (!safe_target) {
        has_recent_target = false;
        planned.control_mode = ControlMode::kHold;
        planned.target_fresh = false;
        planned_target_buffer->Publish(planned);
        publish_trace();
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      double manipulability = 0.0;
      if (use_direct_joint_target) {
        safe_pose = PoseFromJointPositions(model, robot, q_target);
        planned.desired_tcp_pose = safe_pose;
        manipulability = ComputeManipulability(JacobianToEigen(
            model.zeroJacobian(franka::Frame::kEndEffector, q_target, robot.F_T_EE, robot.EE_T_K)));
        ik_ok = true;
      } else {
        ik_ok = SolveIkStep(model,
                            robot,
                            safe_pose,
                            config,
                            planned.control_mode,
                            &q_target,
                            &manipulability);
      }
      if (!ik_ok) {
        planned.faults.ik_rejected = true;
        const bool can_reuse_target =
            has_recent_target && now_ns > last_valid_target_ns &&
            (now_ns - last_valid_target_ns) <= kPlannerTargetGraceNs;
        if (can_reuse_target) {
          planned.control_mode = last_valid_control_mode;
          planned.target_q = last_valid_target_q;
          planned.desired_tcp_pose = last_valid_desired_pose;
          planned.manipulability = last_valid_manipulability;
          planned.target_fresh = true;
        } else {
          planned.control_mode = ControlMode::kHold;
        }
        planned_target_buffer->Publish(planned);
        publish_trace();
        std::this_thread::sleep_for(sleep_period);
        continue;
      }

      planned.target_q = q_target;
      planned.manipulability = manipulability;
      planned.target_fresh = true;
      has_recent_target = true;
      last_valid_target_ns = now_ns;
      last_valid_target_q = q_target;
      last_valid_desired_pose = safe_pose;
      last_valid_control_mode = planned.control_mode;
      last_valid_manipulability = manipulability;
      planned_target_buffer->Publish(planned);
      publish_trace();
      std::this_thread::sleep_for(sleep_period);
    }
  } catch (const std::exception& e) {
    std::cerr << "Planner thread exception: " << e.what() << "\n";
    stop_requested->store(true, std::memory_order_release);
  } catch (...) {
    std::cerr << "Planner thread unknown exception\n";
    stop_requested->store(true, std::memory_order_release);
  }
}

// Simple gripper control loop (rewritten from scratch).
//
// Contract (exactly as the user requested):
//   * State OPEN  + press -> close. If fingers stop moving (object grasped or
//     fully closed), issue stop() and latch state = CLOSE.
//   * State CLOSE + press -> open, move to max width, then latch state = OPEN.
//
// Only two logical states: kOpen and kClose. No preemption, no retry
// escalation, no homing recovery. One action runs until it self-terminates.
void GripperLoopAnalog(franka::Gripper* gripper,
                       const GripperConfig& config,
                       const std::atomic<double>* desired_gripper_width_m,
                       std::atomic<GripperState>* active_gripper_state,
                       std::atomic<double>* measured_gripper_width_m,
                       std::atomic<bool>* stop_requested) {
  try {
    franka::GripperState measured_state = gripper->readOnce();
    measured_gripper_width_m->store(measured_state.width, std::memory_order_release);
    bool closing_stalled = false;
    GripperState current_state = WidthToAnalogState(config, measured_state.width, closing_stalled);
    active_gripper_state->store(current_state, std::memory_order_release);

    auto set_state = [&](GripperState next_state) {
      if (current_state == next_state) {
        return;
      }
      current_state = next_state;
      active_gripper_state->store(current_state, std::memory_order_release);
    };

    // Analog mode needs to respond to fine trigger motions. The XR device already
    // has an effective coarse dead zone in the first half of travel, so do not
    // reuse the binary-mode 2 mm command filter here.
    const double min_delta_m = std::max(0.0, std::min(config.min_command_delta_m, 5e-4));
    const double loop_rate_hz = std::max(config.max_command_rate_hz, 1.0);
    const auto loop_sleep =
        std::chrono::microseconds(static_cast<int64_t>(1e6 / loop_rate_hz));

    while (!stop_requested->load(std::memory_order_acquire)) {
      try {
        measured_state = gripper->readOnce();
      } catch (const std::exception& e) {
        std::cerr << "Gripper analog read failed: " << e.what() << "\n";
        std::this_thread::sleep_for(loop_sleep);
        continue;
      }

      measured_gripper_width_m->store(measured_state.width, std::memory_order_release);
      const double desired_width = ClampWidth(
          config, desired_gripper_width_m->load(std::memory_order_acquire));
      const double error = desired_width - measured_state.width;

      if (closing_stalled) {
        if (error > min_delta_m) {
          closing_stalled = false;
        } else {
          set_state(WidthToAnalogState(config, measured_state.width, true));
          std::this_thread::sleep_for(loop_sleep);
          continue;
        }
      }

      if (std::abs(error) <= min_delta_m) {
        set_state(WidthToAnalogState(config, measured_state.width, false));
        std::this_thread::sleep_for(loop_sleep);
        continue;
      }

      const double command_width = desired_width;
      const bool closing = error < 0.0;
      const bool move_ok = gripper->move(command_width, config.speed_mps);
      measured_state = gripper->readOnce();
      measured_gripper_width_m->store(measured_state.width, std::memory_order_release);

      if (closing) {
        const bool blocked_before_target =
            !move_ok &&
            measured_state.width > (command_width + config.width_tolerance_m);
        if (blocked_before_target) {
          closing_stalled = true;
          set_state(WidthToAnalogState(config, measured_state.width, true));
          std::this_thread::sleep_for(loop_sleep);
          continue;
        }
      }

      set_state(WidthToAnalogState(config, measured_state.width, false));
    }
  } catch (const std::exception& e) {
    std::cerr << "Gripper analog thread exception: " << e.what() << "\n";
    stop_requested->store(true, std::memory_order_release);
  } catch (...) {
    std::cerr << "Gripper analog thread unknown exception\n";
    stop_requested->store(true, std::memory_order_release);
  }
}

void GripperLoop(franka::Gripper* gripper,
                 const GripperConfig& config,
                 const std::atomic<GripperState>* desired_gripper_state,
                 const std::atomic<double>* desired_gripper_width_m,
                 std::atomic<GripperState>* active_gripper_state,
                 std::atomic<double>* measured_gripper_width_m,
                 std::atomic<bool>* stop_requested) {
  if (config.command_mode == GripperCommandMode::kAnalog) {
    GripperLoopAnalog(gripper,
                      config,
                      desired_gripper_width_m,
                      active_gripper_state,
                      measured_gripper_width_m,
                      stop_requested);
    return;
  }
  try {
    franka::GripperState measured_state = gripper->readOnce();
    measured_gripper_width_m->store(measured_state.width, std::memory_order_release);

    // Initial logical state purely from measured width.
    GripperState current_state =
        (measured_state.width >= (config.max_width_m - config.width_tolerance_m))
            ? GripperState::kOpen
            : GripperState::kClose;
    active_gripper_state->store(current_state, std::memory_order_release);

    auto log_state = [](const char* label, GripperState s, double width) {
      const uint64_t t = MonotonicNowNs();
      std::cout << "Gripper " << label << " state=" << ToString(s) << " width=" << width
                << " t_ms=" << (static_cast<double>(t) * 1e-6) << "\n";
    };
    log_state("init", current_state, measured_state.width);

    bool action_in_progress = false;
    GripperState action_target = current_state;
    std::future<bool> action_future;
    double stall_ref_width = measured_state.width;
    uint64_t stall_ref_ns = MonotonicNowNs();
    uint64_t last_read_ok_ns = stall_ref_ns;
    GripperState last_desired_logged = desired_gripper_state->load(std::memory_order_acquire);

    constexpr auto kLoopSleep = std::chrono::milliseconds(10);

    auto set_state = [&](GripperState next_state, const char* reason) {
      if (current_state == next_state) return;
      const uint64_t t = MonotonicNowNs();
      std::cout << "Gripper state " << ToString(current_state) << " -> " << ToString(next_state)
                << " reason=" << reason
                << " t_ms=" << (static_cast<double>(t) * 1e-6) << "\n";
      current_state = next_state;
      active_gripper_state->store(current_state, std::memory_order_release);
    };

    // Waits for the outstanding async move() to finish; swallows exceptions.
    auto drain_future = [&]() {
      if (!action_future.valid()) return;
      try {
        (void)action_future.get();
      } catch (const std::exception& e) {
        std::cerr << "Gripper action ended with exception (ignored): " << e.what() << "\n";
      } catch (...) {
        std::cerr << "Gripper action ended with unknown exception (ignored)\n";
      }
      action_in_progress = false;
    };

    // CLOSE uses franka::Gripper::grasp(). grasp() self-terminates when the
    // fingers contact an object (force threshold in firmware) or reach target
    // width, so we never need to call stop() from another thread (which would
    // violate libfranka's non-thread-safe contract).
    //
    // Large epsilon_outer + tiny epsilon_inner means "grasp succeeds if
    // fingers stop anywhere from 0 to max_width" - we treat any contact or
    // full close as a successful close, matching the user's spec.
    auto start_close = [&]() {
      const uint64_t t = MonotonicNowNs();
      const double target_w = config.min_width_m;
      const double speed = config.speed_mps;
      const double force = config.grasp_force_n;
      const double epsilon_inner = 0.002;  // small tolerance below target
      const double epsilon_outer = config.max_width_m;  // accept any contact above target
      std::cout << "Gripper start_close (grasp) target=" << target_w
                << " force=" << force
                << " speed=" << speed
                << " epsilon_outer=" << epsilon_outer
                << " measured_width=" << measured_state.width
                << " t_ms=" << (static_cast<double>(t) * 1e-6) << "\n";
      action_target = GripperState::kClose;
      stall_ref_width = measured_state.width;
      stall_ref_ns = t;
      action_in_progress = true;
      action_future = std::async(std::launch::async,
                                 [gripper, target_w, speed, force, epsilon_inner, epsilon_outer]() {
                                   return gripper->grasp(target_w, speed, force,
                                                         epsilon_inner, epsilon_outer);
                                 });
    };

    auto start_open = [&]() {
      const uint64_t t = MonotonicNowNs();
      std::cout << "Gripper start_open width_target=" << config.max_width_m
                << " measured_width=" << measured_state.width
                << " t_ms=" << (static_cast<double>(t) * 1e-6) << "\n";
      action_target = GripperState::kOpen;
      stall_ref_width = measured_state.width;
      stall_ref_ns = t;
      action_in_progress = true;
      action_future = std::async(std::launch::async,
                                 [gripper, width = config.max_width_m, speed = config.speed_mps]() {
                                   return gripper->move(width, speed);
                                 });
    };

    while (!stop_requested->load(std::memory_order_acquire)) {
      const uint64_t now_ns = MonotonicNowNs();
      const GripperState desired_state = desired_gripper_state->load(std::memory_order_acquire);

      if (desired_state != last_desired_logged) {
        std::cout << "Gripper desired_state " << ToString(last_desired_logged) << " -> "
                  << ToString(desired_state)
                  << " current=" << ToString(current_state)
                  << " action_in_progress=" << (action_in_progress ? 1 : 0)
                  << " action_target=" << ToString(action_target)
                  << " t_ms=" << (static_cast<double>(now_ns) * 1e-6) << "\n";
        last_desired_logged = desired_state;
      }

      // Read measured gripper state (non-fatal on transient errors).
      try {
        measured_state = gripper->readOnce();
        measured_gripper_width_m->store(measured_state.width, std::memory_order_release);
        last_read_ok_ns = now_ns;
      } catch (const std::exception& e) {
        if (static_cast<double>(now_ns - last_read_ok_ns) * 1e-9 >=
            config.read_failure_timeout_s) {
          std::cerr << "Gripper read failed persistently: " << e.what() << "\n";
        }
        std::this_thread::sleep_for(kLoopSleep);
        continue;
      }

      if (action_in_progress) {
        // 1) Did the async move() finish by itself?
        if (action_future.valid() &&
            action_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
          bool success = false;
          try {
            success = action_future.get();
          } catch (const std::exception& e) {
            std::cerr << "Gripper move() threw: " << e.what() << "\n";
          }
          std::cout << "Gripper action completed target=" << ToString(action_target)
                    << " success=" << (success ? 1 : 0)
                    << " measured_width=" << measured_state.width
                    << " t_ms=" << (static_cast<double>(MonotonicNowNs()) * 1e-6) << "\n";
          action_in_progress = false;
          // Latch state to the action we attempted - user's spec:
          //   "if the gripper stops moving ... set state to closed"
          //   "open ... until full open then set gripper state to open"
          // So on completion (or even on failure), the action_target is what
          // we consider the new state, regardless of move()'s return value.
          set_state(action_target,
                    action_target == GripperState::kOpen ? "open_done" : "close_done");
          std::this_thread::sleep_for(kLoopSleep);
          continue;
        }

        // For close (grasp()): the firmware self-terminates on contact or
        // full close, so no per-tick stall detection is needed. We still
        // enforce a long safety timeout in case grasp() itself misbehaves,
        // so the gripper thread never wedges indefinitely.
        const double kCloseSafetyTimeoutS = 5.0;
        const double kOpenSafetyTimeoutS = 5.0;
        const double elapsed_s = static_cast<double>(now_ns - stall_ref_ns) * 1e-9;
        const double safety_timeout_s = action_target == GripperState::kClose
                                            ? kCloseSafetyTimeoutS
                                            : kOpenSafetyTimeoutS;
        if (elapsed_s >= safety_timeout_s) {
          std::cerr << "Gripper action exceeded safety timeout ("
                    << safety_timeout_s << "s) target=" << ToString(action_target)
                    << " width=" << measured_state.width
                    << "; forcing stop().\n";
          try {
            (void)gripper->stop();
          } catch (const std::exception& e) {
            std::cerr << "Gripper safety stop() failed: " << e.what() << "\n";
          }
          drain_future();
          set_state(action_target,
                    action_target == GripperState::kOpen ? "open_timeout" : "close_timeout");
          std::this_thread::sleep_for(kLoopSleep);
          continue;
        }
      } else {
        // 3) No action running - dispatch if desired differs from current.
        if (desired_state != current_state) {
          if (desired_state == GripperState::kClose) {
            start_close();
          } else if (desired_state == GripperState::kOpen) {
            start_open();
          }
        }
      }

      std::this_thread::sleep_for(kLoopSleep);
    }

    // Shutdown: make sure any outstanding async move() is drained.
    if (action_in_progress) {
      try { (void)gripper->stop(); } catch (...) {}
      drain_future();
    }
  } catch (const std::exception& e) {
    std::cerr << "Gripper thread exception: " << e.what() << "\n";
    stop_requested->store(true, std::memory_order_release);
  } catch (...) {
    std::cerr << "Gripper thread unknown exception\n";
    stop_requested->store(true, std::memory_order_release);
  }
}

}  // namespace

FrankaTeleopController::FrankaTeleopController(const FrankaControllerOptions& options,
                                               const TeleopBridgeConfig& config,
                                               const LatestCommandBuffer* command_buffer,
                                               const LatestPolicyActionBuffer* policy_action_buffer,
                                               LatestObservationBuffer* observation_buffer)
    : options_(options),
      config_(config),
      command_buffer_(command_buffer),
      policy_action_buffer_(policy_action_buffer),
      observation_buffer_(observation_buffer) {}

int FrankaTeleopController::Run(std::atomic<bool>* stop_requested) {
  std::thread planner_thread;
  std::thread gripper_thread;
  std::unique_ptr<TraceRecorder> trace_recorder;
  auto join_threads = [&]() {
    stop_requested->store(true, std::memory_order_release);
    if (planner_thread.joinable()) {
      planner_thread.join();
    }
    if (gripper_thread.joinable()) {
      gripper_thread.join();
    }
  };
  auto stop_trace = [&]() {
    if (trace_recorder == nullptr) {
      return;
    }
    trace_recorder->Stop();
    std::cout << "Trace capture stopped. output_dir=" << trace_recorder->output_dir()
              << " dropped_planner=" << trace_recorder->dropped_planner()
              << " dropped_rt=" << trace_recorder->dropped_rt() << "\n";
    trace_recorder.reset();
  };

  try {
    if (options_.trace.enabled) {
      TraceConfig trace_config{};
      trace_config.enabled = true;
      trace_config.output_dir = options_.trace.output_dir;
      trace_config.planner_decimation = std::max<uint32_t>(1, options_.trace.planner_decimation);
      trace_config.rt_decimation = std::max<uint32_t>(1, options_.trace.rt_decimation);

      trace_recorder = std::make_unique<TraceRecorder>(trace_config);
      std::string trace_error;
      if (!trace_recorder->Start(&trace_error)) {
        std::cerr << "Failed to start trace recorder: " << trace_error << "\n";
        return 8;
      }
      std::cout << "Trace capture enabled: output_dir=" << trace_recorder->output_dir()
                << " planner_decimation=" << trace_config.planner_decimation
                << " rt_decimation=" << trace_config.rt_decimation << "\n";
    }

    franka::Robot robot(options_.robot_ip);
    ConfigureConservativeBehavior(&robot);
    robot.setLoad(config_.load.mass_kg, config_.load.center_of_mass_m, config_.load.inertia_kgm2);

    std::unique_ptr<franka::Gripper> gripper;
    std::atomic<double> measured_gripper_width_m{0.0};
    if (config_.gripper.enabled) {
      try {
        gripper = std::make_unique<franka::Gripper>(options_.robot_ip);
        franka::GripperState state = gripper->readOnce();
        if (state.max_width <= 1e-4) {
          std::cout << "Gripper max_width unavailable, running homing...\n";
          if (!gripper->homing()) {
            std::cerr << "Gripper homing returned false.\n";
          }
          state = gripper->readOnce();
        }
        if (state.max_width > 1e-4) {
          config_.gripper.max_width_m = std::min(config_.gripper.max_width_m, state.max_width);
        }
        measured_gripper_width_m.store(state.width, std::memory_order_release);
        std::cout << "Gripper ready: width=" << state.width
                  << " max_width=" << state.max_width
                  << " command_mode="
                  << (config_.gripper.command_mode == GripperCommandMode::kBinary ? "binary"
                                                                                  : "analog")
                  << "\n";
      } catch (const std::exception& e) {
        std::cerr << "Gripper unavailable, continuing without gripper control: " << e.what() << "\n";
        gripper.reset();
      }
    }

    if (!RecoverRobotIfNeeded(&robot)) {
      std::cerr << "Failed to recover robot from reflex mode.\n";
      return 5;
    }
    franka::Model model = robot.loadModel();

    if (!MoveToSafeHomeRoute(&robot, config_, stop_requested)) {
      std::cerr << "Home move interrupted.\n";
      return 6;
    }
    if (gripper != nullptr) {
      CommandGripperOpen(gripper.get(), config_.gripper, &measured_gripper_width_m);
    }

    LatestRobotStateBuffer robot_state_buffer;
    LatestPlannedTargetBuffer planned_target_buffer;

    const RobotSnapshot initial_robot_snapshot = ToSnapshot(robot.readOnce());
    robot_state_buffer.Publish(initial_robot_snapshot);

    PublishHoldTarget(
        initial_robot_snapshot, config_.gripper.max_width_m, TeleopState::kDisconnected, &planned_target_buffer);

    std::atomic<GripperState> desired_gripper_state{GripperState::kOpen};
    std::atomic<double> desired_gripper_width_m{config_.gripper.max_width_m};
    std::atomic<uint64_t> requested_rehome_request_id{0};
    std::atomic<uint64_t> completed_rehome_request_id{0};
    std::atomic<bool> rehome_in_progress{false};
    const GripperState initial_gripper_state = !config_.gripper.enabled
                                                   ? GripperState::kOpen
                                                   : (gripper == nullptr
                                                          ? GripperState::kFault
                                                          : (measured_gripper_width_m.load(
                                                                     std::memory_order_acquire) >=
                                                                     (config_.gripper.max_width_m -
                                                                      config_.gripper.width_tolerance_m)
                                                                 ? GripperState::kOpen
                                                                 : GripperState::kClose));
    std::atomic<GripperState> active_gripper_state{initial_gripper_state};
    planner_thread = std::thread(PlannerLoop,
                                 std::cref(config_),
                                 std::cref(model),
                                 command_buffer_,
                                 policy_action_buffer_,
                                 &robot_state_buffer,
                                 &planned_target_buffer,
                                 trace_recorder.get(),
                                 std::max<uint32_t>(1, options_.trace.planner_decimation),
                                 &active_gripper_state,
                                 &desired_gripper_state,
                                 &desired_gripper_width_m,
                                 &requested_rehome_request_id,
                                 &completed_rehome_request_id,
                                 &rehome_in_progress,
                                 stop_requested);

    if (gripper != nullptr && config_.allow_motion) {
      std::cout << "Gripper control thread started.\n";
      gripper_thread = std::thread(GripperLoop,
                                   gripper.get(),
                                   std::cref(config_.gripper),
                                   &desired_gripper_state,
                                   &desired_gripper_width_m,
                                   &active_gripper_state,
                                   &measured_gripper_width_m,
                                   stop_requested);
    } else if (gripper != nullptr) {
      std::cout << "Gripper control disabled because allow_motion=false.\n";
    }

    uint64_t rt_last_ns = 0;
    uint64_t rt_trace_counter = 0;
    uint64_t last_success_rate_log_ns = 0;
    uint64_t motion_inhibit_until_ns = 0;
    uint64_t policy_tracking_fault_until_ns = 0;
    uint64_t policy_tracking_fault_accum_ns = 0;
    uint32_t cartesian_contact_cycles = 0;
    uint64_t last_completed_policy_rehome_request_id = 0;
    uint64_t pending_policy_rehome_request_id = 0;
    JointPositionTrajectoryGenerator trajectory_generator(config_);
    const uint32_t rt_trace_decimation = std::max<uint32_t>(1, options_.trace.rt_decimation);
    while (!stop_requested->load(std::memory_order_acquire)) {
      try {
        uint64_t pending_rehome_request_id = 0;
        uint32_t rehome_settled_cycles = 0;
        robot.control([&](const franka::RobotState& state,
                          franka::Duration period) -> franka::JointPositions {
          const uint64_t now_ns = MonotonicNowNs();
          const uint64_t rt_loop_dt_ns = (rt_last_ns == 0) ? 0 : (now_ns - rt_last_ns);
          rt_last_ns = now_ns;
          if (last_success_rate_log_ns == 0 || (now_ns - last_success_rate_log_ns) >= 5000000000ull) {
            last_success_rate_log_ns = now_ns;
            std::cerr << "libfranka control_command_success_rate=" << state.control_command_success_rate
                      << " q_err_max=" << [&state]() {
                           double max_err = 0.0;
                           for (size_t i = 0; i < 7; ++i) {
                             max_err = std::max(max_err, std::abs(state.q[i] - state.q_d[i]));
                           }
                           return max_err;
                         }()
                      << "\n";
          }
          const RobotSnapshot robot_snapshot = ToSnapshot(state);
          robot_state_buffer.Publish(robot_snapshot);
          if (config_.control_source == ControlSource::kPolicy && policy_action_buffer_ != nullptr) {
            const PolicyActionCommand policy_cmd = policy_action_buffer_->ReadLatest();
            const bool new_rehome_request =
                policy_cmd.request_rehome && policy_cmd.operator_request_id != 0 &&
                policy_cmd.operator_request_id > last_completed_policy_rehome_request_id;
            if (new_rehome_request) {
              pending_policy_rehome_request_id = policy_cmd.operator_request_id;
            }
          }

          const bool collision_active =
              AnyPositive(state.joint_collision) || AnyPositive(state.cartesian_collision);
          const bool cartesian_contact_active = AnyPositive(state.cartesian_contact);
          if (cartesian_contact_active) {
            cartesian_contact_cycles =
                std::min<uint32_t>(cartesian_contact_cycles + 1, kCartesianContactDebounceCycles);
          } else {
            cartesian_contact_cycles = 0;
          }
          const bool sustained_cartesian_contact =
              cartesian_contact_cycles >= kCartesianContactDebounceCycles;

          const PlannedTarget planned = planned_target_buffer.ReadLatest();
          const double dt = std::max(period.toSec(), 1e-6);
          double max_measured_command_error = 0.0;
          double max_command_target_error = 0.0;
          for (size_t i = 0; i < 7; ++i) {
            max_measured_command_error =
                std::max(max_measured_command_error, std::abs(state.q[i] - state.q_d[i]));
            max_command_target_error =
                std::max(max_command_target_error, std::abs(planned.target_q[i] - state.q_d[i]));
          }
          const bool policy_tracking_error =
              config_.control_source == ControlSource::kPolicy && planned.teleop_active &&
              (max_measured_command_error > config_.policy.tracking_position_error_rad ||
               max_command_target_error > config_.policy.tracking_target_error_rad);
          if (policy_tracking_error) {
            policy_tracking_fault_accum_ns += static_cast<uint64_t>(dt * 1e9);
          } else {
            policy_tracking_fault_accum_ns = 0;
          }
          if (policy_tracking_fault_accum_ns >=
              static_cast<uint64_t>(config_.policy.tracking_fault_dwell_s * 1e9)) {
            policy_tracking_fault_until_ns =
                std::max(policy_tracking_fault_until_ns,
                         now_ns + static_cast<uint64_t>(config_.policy.tracking_inhibit_s * 1e9));
            policy_tracking_fault_accum_ns = 0;
          }
          const bool recovery_inhibit_active = now_ns < motion_inhibit_until_ns;
          const bool policy_tracking_inhibit_active = now_ns < policy_tracking_fault_until_ns;
          const uint64_t requested_rehome_id =
              requested_rehome_request_id.load(std::memory_order_acquire);
          const uint64_t completed_rehome_id =
              completed_rehome_request_id.load(std::memory_order_acquire);
          if (pending_rehome_request_id == 0 && requested_rehome_id > completed_rehome_id) {
            pending_rehome_request_id = requested_rehome_id;
          }
          const bool rehome_pending =
              pending_rehome_request_id != 0 || pending_policy_rehome_request_id != 0;
          const bool apply_motion =
              config_.allow_motion && !rehome_pending && planned.teleop_active &&
              planned.target_fresh && planned.control_mode != ControlMode::kHold &&
              !collision_active && !sustained_cartesian_contact &&
              !recovery_inhibit_active &&
              !policy_tracking_inhibit_active;

          const double max_step = std::min(config_.ik.max_joint_step_rad,
                                           config_.ik.max_joint_velocity_radps * dt);
          const double rt_alpha = 0.0;
          std::array<double, 7> target_delta{};
          std::array<double, 7> filtered_delta{};
          std::array<double, 7> command_delta{};
          std::array<uint8_t, 7> clamp_saturated{};
          std::array<double, 7> q_cmd{};
          if (rehome_pending) {
            q_cmd = state.q_d;
            target_delta.fill(0.0);
            filtered_delta.fill(0.0);
            command_delta.fill(0.0);
            clamp_saturated.fill(0);
          } else {
            q_cmd = trajectory_generator.Update(planned.target_q,
                                                state.q_d,
                                                dt,
                                                apply_motion,
                                                &target_delta,
                                                &filtered_delta,
                                                &command_delta,
                                                &clamp_saturated);
          }
          double max_abs_target_delta = 0.0;
          double max_abs_filtered_delta = 0.0;
          double max_abs_command_delta = 0.0;
          for (size_t i = 0; i < 7; ++i) {
            max_abs_target_delta = std::max(max_abs_target_delta, std::abs(target_delta[i]));
            max_abs_filtered_delta = std::max(max_abs_filtered_delta, std::abs(filtered_delta[i]));
            max_abs_command_delta = std::max(max_abs_command_delta, std::abs(command_delta[i]));
          }
          const Pose commanded_target_pose = MatrixToPose(
              model.pose(franka::Frame::kEndEffector, q_cmd, state.F_T_EE, state.EE_T_K));

          RobotObservation obs{};
          obs.timestamp_ns = now_ns;
          obs.q = state.q;
          obs.q_cmd = q_cmd;
          obs.dq = state.dq;
          obs.tcp_pose = MatrixToPose(state.O_T_EE);
          obs.desired_target_tcp_pose = planned.desired_tcp_pose;
          obs.commanded_target_tcp_pose = commanded_target_pose;
          obs.gripper_width = measured_gripper_width_m.load(std::memory_order_acquire);
          obs.gripper_state = active_gripper_state.load(std::memory_order_acquire);
          obs.executed_action = planned.requested_action;
          obs.control_mode = planned.teleop_active ? planned.control_mode : ControlMode::kHold;
          obs.teleop_state = planned.teleop_state;
          obs.packet_age_ns = planned.packet_age_ns;
          obs.target_age_ns =
              now_ns > planned.target_timestamp_ns ? (now_ns - planned.target_timestamp_ns) : 0;
          obs.target_fresh = planned.target_fresh;
          obs.teleop_active = planned.teleop_active;
          obs.episode_start = planned.episode_start;
          obs.episode_end = planned.episode_end;
          obs.target_manipulability = planned.manipulability;
          obs.faults = planned.faults;
          obs.faults.robot_not_ready = obs.faults.robot_not_ready || !robot_snapshot.robot_ok;
          obs.faults.gripper_fault = obs.faults.gripper_fault || obs.gripper_state == GripperState::kFault;
          obs.faults.control_exception =
              obs.faults.control_exception || collision_active || sustained_cartesian_contact ||
              recovery_inhibit_active || policy_tracking_inhibit_active;
          observation_buffer_->Publish(obs);

          if (trace_recorder != nullptr && ((rt_trace_counter++ % rt_trace_decimation) == 0)) {
            RtTraceSample trace{};
            trace.timestamp_ns = now_ns;
            trace.loop_dt_ns = rt_loop_dt_ns;
            trace.callback_period_ns = static_cast<uint64_t>(dt * 1e9);
            trace.target_age_ns = obs.target_age_ns;
            trace.teleop_state = static_cast<int>(obs.teleop_state);
            trace.control_mode = static_cast<int>(obs.control_mode);
            trace.teleop_active = obs.teleop_active;
            trace.target_fresh = obs.target_fresh;
            trace.apply_motion = apply_motion;
            trace.faults = obs.faults;
            trace.max_step = max_step;
            trace.rt_alpha = rt_alpha;
            trace.q = state.q;
            trace.dq = state.dq;
            trace.q_d = state.q_d;
            trace.q_planned = planned.target_q;
            trace.q_cmd = q_cmd;
            trace.target_delta = target_delta;
            trace.filtered_delta = filtered_delta;
            trace.command_delta = command_delta;
            trace.clamp_saturated = clamp_saturated;
            trace.max_abs_target_delta = max_abs_target_delta;
            trace.max_abs_filtered_delta = max_abs_filtered_delta;
            trace.max_abs_command_delta = max_abs_command_delta;
            trace_recorder->PushRt(trace);
          }

          franka::JointPositions out(q_cmd);
          if (rehome_pending && !apply_motion) {
            bool settled = max_abs_command_delta <= kRehomeStopCommandToleranceRad;
            for (size_t i = 0; i < 7 && settled; ++i) {
              settled = settled &&
                        std::abs(state.dq[i]) <= kRehomeStopVelocityToleranceRadPerS &&
                        std::abs(state.dq_d[i]) <= kRehomeStopVelocityToleranceRadPerS;
            }
            if (settled) {
              ++rehome_settled_cycles;
            } else {
              rehome_settled_cycles = 0;
            }
            if (rehome_settled_cycles >= kRehomeStopSettledCycles) {
              return franka::MotionFinished(out);
            }
          } else {
            rehome_settled_cycles = 0;
          }
          if (stop_requested->load(std::memory_order_acquire)) {
            return franka::MotionFinished(out);
          }
          return out;
        },
                      franka::ControllerMode::kJointImpedance,
                      config_.limit_rate,
                      config_.lpf_cutoff_frequency);
        if (stop_requested->load(std::memory_order_acquire)) {
          break;
        }
        if (pending_policy_rehome_request_id != 0) {
          const uint64_t request_id = pending_policy_rehome_request_id;
          pending_policy_rehome_request_id = 0;
          std::cerr << "Processing policy re-home request id=" << request_id << "\n";
          if (!RecoverRobotIfNeeded(&robot)) {
            std::cerr << "Failed to recover robot before re-home.\n";
            return 5;
          }
          trajectory_generator.Reset();
          rt_last_ns = 0;
          last_success_rate_log_ns = 0;
          cartesian_contact_cycles = 0;
          if (!MoveToSafeHomeRoute(&robot, config_, stop_requested)) {
            if (stop_requested->load(std::memory_order_acquire)) {
              break;
            }
            std::cerr << "Runtime re-home interrupted.\n";
            return 6;
          }
          desired_gripper_state.store(GripperState::kOpen, std::memory_order_release);
          desired_gripper_width_m.store(config_.gripper.max_width_m, std::memory_order_release);
          last_completed_policy_rehome_request_id = request_id;
          motion_inhibit_until_ns = MonotonicNowNs() + kPostRehomeMotionInhibitNs;
          const RobotSnapshot home_snapshot = ToSnapshot(robot.readOnce());
          robot_state_buffer.Publish(home_snapshot);
          PlannedTarget hold_target = planned_target_buffer.ReadLatest();
          hold_target.target_timestamp_ns = MonotonicNowNs();
          hold_target.target_q = home_snapshot.q;
          hold_target.desired_tcp_pose = home_snapshot.tcp_pose;
          hold_target.control_mode = ControlMode::kHold;
          hold_target.teleop_active = false;
          hold_target.target_fresh = false;
          hold_target.requested_action = TeleopAction{};
          planned_target_buffer.Publish(hold_target);
          std::cerr << "Completed policy re-home request id=" << request_id << "\n";
          continue;
        }
        if (pending_rehome_request_id != 0) {
          rehome_in_progress.store(true, std::memory_order_release);
          if (!RecoverRobotIfNeeded(&robot)) {
            std::cerr << "Failed to recover robot before rehome.\n";
            return 5;
          }
          trajectory_generator.Reset();
          rt_last_ns = 0;
          last_success_rate_log_ns = 0;
          cartesian_contact_cycles = 0;
          if (!MoveToSafeHomeRoute(&robot, config_, stop_requested)) {
            rehome_in_progress.store(false, std::memory_order_release);
            if (stop_requested->load(std::memory_order_acquire)) {
              break;
            }
            std::cerr << "Runtime rehome interrupted.\n";
            return 6;
          }
          desired_gripper_state.store(GripperState::kOpen, std::memory_order_release);
          desired_gripper_width_m.store(config_.gripper.max_width_m, std::memory_order_release);
          const RobotSnapshot home_snapshot = ToSnapshot(robot.readOnce());
          robot_state_buffer.Publish(home_snapshot);
          completed_rehome_request_id.store(pending_rehome_request_id, std::memory_order_release);
          motion_inhibit_until_ns = MonotonicNowNs() + kPostRehomeMotionInhibitNs;
          PublishHoldTarget(home_snapshot,
                            config_.gripper.max_width_m,
                            TeleopState::kConnectedIdle,
                            &planned_target_buffer);
          rehome_in_progress.store(false, std::memory_order_release);
          continue;
        }
        break;
      } catch (const franka::ControlException& e) {
        const std::string what = e.what();
        std::cerr << "Franka control exception: " << what << "\n";
        if (what.find("reflex") == std::string::npos) {
          throw;
        }
        try {
          robot.automaticErrorRecovery();
          trajectory_generator.Reset();
          motion_inhibit_until_ns = MonotonicNowNs() + kPostRehomeMotionInhibitNs;
          std::cerr << "Recovered from reflex; continuing teleop after brief hold.\n";
        } catch (const std::exception& recovery_error) {
          std::cerr << "Automatic error recovery failed: " << recovery_error.what() << "\n";
          throw;
        }
      }
    }

    join_threads();
    stop_trace();
    return 0;
  } catch (const franka::ControlException& e) {
    join_threads();
    stop_trace();
    std::cerr << "Franka control exception: " << e.what() << "\n";
    return 2;
  } catch (const franka::Exception& e) {
    join_threads();
    stop_trace();
    std::cerr << "Franka exception: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    join_threads();
    stop_trace();
    std::cerr << "Std exception: " << e.what() << "\n";
    return 4;
  } catch (...) {
    join_threads();
    stop_trace();
    std::cerr << "Unknown controller exception\n";
    return 7;
  }
}

}  // namespace teleop
