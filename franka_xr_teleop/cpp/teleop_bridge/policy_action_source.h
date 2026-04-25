#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "common_types.h"

namespace teleop {

class PolicyActionSource {
 public:
  PolicyActionSource(std::string bind_ip,
                     uint16_t action_port,
                     LatestPolicyActionBuffer* action_buffer,
                     std::atomic<bool>* stop_requested);
  ~PolicyActionSource();

  bool Start();
  void Stop();

  uint64_t last_packet_time_ns() const {
    return last_packet_time_ns_.load(std::memory_order_acquire);
  }
  uint64_t received_count() const { return received_count_.load(std::memory_order_acquire); }
  uint64_t dropped_count() const { return dropped_count_.load(std::memory_order_acquire); }

 private:
  void Run();

  std::string bind_ip_;
  uint16_t action_port_ = 0;
  LatestPolicyActionBuffer* action_buffer_;
  std::atomic<bool>* stop_requested_;

  std::atomic<bool> running_{false};
  std::thread thread_;
  int sock_ = -1;

  std::atomic<uint64_t> sequence_id_{0};
  std::atomic<uint64_t> last_packet_time_ns_{0};
  std::atomic<uint64_t> received_count_{0};
  std::atomic<uint64_t> dropped_count_{0};
};

}  // namespace teleop
