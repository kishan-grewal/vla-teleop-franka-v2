#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>

#include "common_types.h"

namespace teleop {

struct TraceConfig {
  bool enabled = false;
  std::string output_dir;
  uint32_t planner_decimation = 1;
  uint32_t rt_decimation = 1;
};

struct PlannerTraceSample {
  uint64_t timestamp_ns = 0;
  uint64_t loop_dt_ns = 0;
  uint64_t xr_timestamp_ns = 0;
  uint64_t xr_sequence_id = 0;
  uint64_t packet_age_ns = 0;
  int teleop_state = 0;
  int control_mode = 0;
  bool teleop_active = false;
  bool target_fresh = false;
  bool deadman_latched = false;
  bool has_target = false;
  bool safe_target = false;
  bool ik_ok = false;
  FaultFlags faults{};
  double control_trigger_value = 0.0;
  std::array<double, 3> xr_position{};
  std::array<double, 3> desired_position{};
  std::array<double, 3> safe_position{};
  std::array<double, 3> robot_position{};
  std::array<double, 3> requested_delta_translation{};
  std::array<double, 3> requested_delta_rotation{};
  std::array<double, 3> safe_delta_translation{};
  std::array<double, 3> safe_delta_rotation{};
  std::array<double, 7> q_robot{};
  std::array<double, 7> q_raw_target{};
  std::array<double, 7> q_planned{};
  double manipulability = 0.0;
};

struct RtTraceSample {
  uint64_t timestamp_ns = 0;
  uint64_t loop_dt_ns = 0;
  uint64_t callback_period_ns = 0;
  uint64_t target_age_ns = 0;
  int teleop_state = 0;
  int control_mode = 0;
  bool teleop_active = false;
  bool target_fresh = false;
  bool apply_motion = false;
  FaultFlags faults{};
  double max_step = 0.0;
  double rt_alpha = 0.0;
  std::array<double, 7> q{};
  std::array<double, 7> dq{};
  std::array<double, 7> q_d{};
  std::array<double, 7> q_planned{};
  std::array<double, 7> q_cmd{};
  std::array<double, 7> target_delta{};
  std::array<double, 7> filtered_delta{};
  std::array<double, 7> command_delta{};
  std::array<uint8_t, 7> clamp_saturated{};
  double max_abs_target_delta = 0.0;
  double max_abs_filtered_delta = 0.0;
  double max_abs_command_delta = 0.0;
};

class TraceRecorder {
 public:
  explicit TraceRecorder(const TraceConfig& config);
  ~TraceRecorder();

  bool Start(std::string* error);
  void Stop();

  void PushPlanner(const PlannerTraceSample& sample);
  void PushRt(const RtTraceSample& sample);

  uint64_t dropped_planner() const { return dropped_planner_.load(std::memory_order_relaxed); }
  uint64_t dropped_rt() const { return dropped_rt_.load(std::memory_order_relaxed); }
  const std::string& output_dir() const { return output_dir_; }

 private:
  template <typename T, size_t Capacity>
  class SpscQueue {
   public:
    bool Push(const T& value) {
      const size_t head = head_.load(std::memory_order_relaxed);
      const size_t tail = tail_.load(std::memory_order_acquire);
      if (head - tail >= Capacity) {
        return false;
      }
      values_[head % Capacity] = value;
      head_.store(head + 1, std::memory_order_release);
      return true;
    }

    bool Pop(T* out) {
      const size_t tail = tail_.load(std::memory_order_relaxed);
      const size_t head = head_.load(std::memory_order_acquire);
      if (tail == head) {
        return false;
      }
      *out = values_[tail % Capacity];
      tail_.store(tail + 1, std::memory_order_release);
      return true;
    }

    bool Empty() const {
      const size_t tail = tail_.load(std::memory_order_acquire);
      const size_t head = head_.load(std::memory_order_acquire);
      return tail == head;
    }

   private:
    std::array<T, Capacity> values_{};
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
  };

  void WriterLoop();
  void WritePlannerHeader();
  void WriteRtHeader();
  void WritePlannerRow(const PlannerTraceSample& sample);
  void WriteRtRow(const RtTraceSample& sample);

  TraceConfig config_{};
  std::string output_dir_{};
  std::thread writer_thread_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<uint64_t> dropped_planner_{0};
  std::atomic<uint64_t> dropped_rt_{0};

  SpscQueue<PlannerTraceSample, 8192> planner_queue_{};
  SpscQueue<RtTraceSample, 32768> rt_queue_{};

  std::ofstream planner_file_{};
  std::ofstream rt_file_{};
};

}  // namespace teleop
