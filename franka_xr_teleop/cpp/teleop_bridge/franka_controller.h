#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "common_types.h"

namespace teleop {

struct TraceOptions {
  bool enabled = false;
  std::string output_dir;
  uint32_t planner_decimation = 1;
  uint32_t rt_decimation = 1;
};

struct FrankaControllerOptions {
  std::string robot_ip;
  TraceOptions trace{};
};

class FrankaTeleopController {
 public:
  FrankaTeleopController(const FrankaControllerOptions& options,
                         const TeleopBridgeConfig& config,
                         const LatestCommandBuffer* command_buffer,
                         const LatestPolicyActionBuffer* policy_action_buffer,
                         LatestObservationBuffer* observation_buffer);

  // Blocking call. Returns 0 on clean stop, non-zero on error.
  int Run(std::atomic<bool>* stop_requested);

 private:
  FrankaControllerOptions options_{};
  TeleopBridgeConfig config_{};
  const LatestCommandBuffer* command_buffer_;
  const LatestPolicyActionBuffer* policy_action_buffer_;
  LatestObservationBuffer* observation_buffer_;
};

}  // namespace teleop
