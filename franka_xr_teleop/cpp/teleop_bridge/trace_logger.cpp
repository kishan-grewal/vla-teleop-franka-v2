#include "trace_logger.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace teleop {
namespace {

template <size_t N>
void WriteDoubleArray(std::ofstream* file, const std::array<double, N>& values) {
  for (size_t i = 0; i < N; ++i) {
    *file << ',' << values[i];
  }
}

template <size_t N>
void WriteUint8Array(std::ofstream* file, const std::array<uint8_t, N>& values) {
  for (size_t i = 0; i < N; ++i) {
    *file << ',' << static_cast<int>(values[i]);
  }
}

int BoolToInt(bool value) {
  return value ? 1 : 0;
}

}  // namespace

TraceRecorder::TraceRecorder(const TraceConfig& config) : config_(config) {}

TraceRecorder::~TraceRecorder() {
  Stop();
}

bool TraceRecorder::Start(std::string* error) {
  if (!config_.enabled) {
    return true;
  }

  if (started_.load(std::memory_order_acquire)) {
    return true;
  }

  try {
    std::filesystem::create_directories(config_.output_dir);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("Failed to create trace directory: ") + e.what();
    }
    return false;
  }

  output_dir_ = config_.output_dir;
  const std::filesystem::path planner_path =
      std::filesystem::path(output_dir_) / "planner_trace.csv";
  const std::filesystem::path rt_path =
      std::filesystem::path(output_dir_) / "rt_trace.csv";

  planner_file_.open(planner_path.string(), std::ios::out | std::ios::trunc);
  rt_file_.open(rt_path.string(), std::ios::out | std::ios::trunc);
  if (!planner_file_.is_open() || !rt_file_.is_open()) {
    if (error != nullptr) {
      *error = "Failed to open trace output files";
    }
    return false;
  }

  planner_file_ << std::fixed << std::setprecision(9);
  rt_file_ << std::fixed << std::setprecision(9);
  WritePlannerHeader();
  WriteRtHeader();

  stop_requested_.store(false, std::memory_order_release);
  started_.store(true, std::memory_order_release);
  writer_thread_ = std::thread(&TraceRecorder::WriterLoop, this);
  return true;
}

void TraceRecorder::Stop() {
  if (!started_.load(std::memory_order_acquire)) {
    return;
  }
  stop_requested_.store(true, std::memory_order_release);
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
  if (planner_file_.is_open()) {
    planner_file_.flush();
    planner_file_.close();
  }
  if (rt_file_.is_open()) {
    rt_file_.flush();
    rt_file_.close();
  }
  started_.store(false, std::memory_order_release);
}

void TraceRecorder::PushPlanner(const PlannerTraceSample& sample) {
  if (!started_.load(std::memory_order_acquire)) {
    return;
  }
  if (!planner_queue_.Push(sample)) {
    dropped_planner_.fetch_add(1, std::memory_order_relaxed);
  }
}

void TraceRecorder::PushRt(const RtTraceSample& sample) {
  if (!started_.load(std::memory_order_acquire)) {
    return;
  }
  if (!rt_queue_.Push(sample)) {
    dropped_rt_.fetch_add(1, std::memory_order_relaxed);
  }
}

void TraceRecorder::WriterLoop() {
  while (!stop_requested_.load(std::memory_order_acquire) || !planner_queue_.Empty() ||
         !rt_queue_.Empty()) {
    bool wrote_any = false;

    PlannerTraceSample planner_sample{};
    while (planner_queue_.Pop(&planner_sample)) {
      WritePlannerRow(planner_sample);
      wrote_any = true;
    }

    RtTraceSample rt_sample{};
    while (rt_queue_.Pop(&rt_sample)) {
      WriteRtRow(rt_sample);
      wrote_any = true;
    }

    if (!wrote_any) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void TraceRecorder::WritePlannerHeader() {
  planner_file_
      << "timestamp_ns,loop_dt_ns,xr_timestamp_ns,xr_sequence_id,packet_age_ns,teleop_state,"
         "control_mode,teleop_active,target_fresh,deadman_latched,has_target,safe_target,ik_ok,"
         "fault_packet_timeout,fault_jump_rejected,fault_workspace_clamped,fault_robot_not_ready,"
         "fault_ik_rejected,control_trigger_value";

  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",xr_pos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",desired_pos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",safe_pos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",robot_pos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",requested_dpos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",requested_drot_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",safe_dpos_" << i;
  }
  for (size_t i = 0; i < 3; ++i) {
    planner_file_ << ",safe_drot_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    planner_file_ << ",q_robot_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    planner_file_ << ",q_raw_target_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    planner_file_ << ",q_planned_" << i;
  }
  planner_file_ << ",manipulability\n";
}

void TraceRecorder::WriteRtHeader() {
  rt_file_ << "timestamp_ns,loop_dt_ns,callback_period_ns,target_age_ns,teleop_state,control_mode,"
              "teleop_active,target_fresh,apply_motion,fault_packet_timeout,fault_jump_rejected,"
              "fault_workspace_clamped,fault_robot_not_ready,fault_ik_rejected,max_step,rt_alpha";

  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",q_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",dq_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",q_d_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",q_planned_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",q_cmd_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",target_delta_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",filtered_delta_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",command_delta_" << i;
  }
  for (size_t i = 0; i < 7; ++i) {
    rt_file_ << ",clamp_saturated_" << i;
  }
  rt_file_ << ",max_abs_target_delta,max_abs_filtered_delta,max_abs_command_delta\n";
}

void TraceRecorder::WritePlannerRow(const PlannerTraceSample& sample) {
  planner_file_ << sample.timestamp_ns << ',' << sample.loop_dt_ns << ',' << sample.xr_timestamp_ns
                << ',' << sample.xr_sequence_id << ',' << sample.packet_age_ns << ','
                << sample.teleop_state << ',' << sample.control_mode << ','
                << BoolToInt(sample.teleop_active) << ',' << BoolToInt(sample.target_fresh) << ','
                << BoolToInt(sample.deadman_latched) << ',' << BoolToInt(sample.has_target) << ','
                << BoolToInt(sample.safe_target) << ',' << BoolToInt(sample.ik_ok) << ','
                << BoolToInt(sample.faults.packet_timeout) << ','
                << BoolToInt(sample.faults.jump_rejected) << ','
                << BoolToInt(sample.faults.workspace_clamped) << ','
                << BoolToInt(sample.faults.robot_not_ready) << ','
                << BoolToInt(sample.faults.ik_rejected) << ',' << sample.control_trigger_value;

  WriteDoubleArray(&planner_file_, sample.xr_position);
  WriteDoubleArray(&planner_file_, sample.desired_position);
  WriteDoubleArray(&planner_file_, sample.safe_position);
  WriteDoubleArray(&planner_file_, sample.robot_position);
  WriteDoubleArray(&planner_file_, sample.requested_delta_translation);
  WriteDoubleArray(&planner_file_, sample.requested_delta_rotation);
  WriteDoubleArray(&planner_file_, sample.safe_delta_translation);
  WriteDoubleArray(&planner_file_, sample.safe_delta_rotation);
  WriteDoubleArray(&planner_file_, sample.q_robot);
  WriteDoubleArray(&planner_file_, sample.q_raw_target);
  WriteDoubleArray(&planner_file_, sample.q_planned);
  planner_file_ << ',' << sample.manipulability << '\n';
}

void TraceRecorder::WriteRtRow(const RtTraceSample& sample) {
  rt_file_ << sample.timestamp_ns << ',' << sample.loop_dt_ns << ',' << sample.callback_period_ns
           << ',' << sample.target_age_ns << ',' << sample.teleop_state << ','
           << sample.control_mode << ',' << BoolToInt(sample.teleop_active) << ','
           << BoolToInt(sample.target_fresh) << ',' << BoolToInt(sample.apply_motion) << ','
           << BoolToInt(sample.faults.packet_timeout) << ','
           << BoolToInt(sample.faults.jump_rejected) << ','
           << BoolToInt(sample.faults.workspace_clamped) << ','
           << BoolToInt(sample.faults.robot_not_ready) << ','
           << BoolToInt(sample.faults.ik_rejected) << ',' << sample.max_step << ','
           << sample.rt_alpha;

  WriteDoubleArray(&rt_file_, sample.q);
  WriteDoubleArray(&rt_file_, sample.dq);
  WriteDoubleArray(&rt_file_, sample.q_d);
  WriteDoubleArray(&rt_file_, sample.q_planned);
  WriteDoubleArray(&rt_file_, sample.q_cmd);
  WriteDoubleArray(&rt_file_, sample.target_delta);
  WriteDoubleArray(&rt_file_, sample.filtered_delta);
  WriteDoubleArray(&rt_file_, sample.command_delta);
  WriteUint8Array(&rt_file_, sample.clamp_saturated);
  rt_file_ << ',' << sample.max_abs_target_delta << ',' << sample.max_abs_filtered_delta << ','
           << sample.max_abs_command_delta << '\n';
}

}  // namespace teleop
