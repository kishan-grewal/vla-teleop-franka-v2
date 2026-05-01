#include "policy_action_source.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "math_utils.h"

namespace teleop {
namespace {

using json = nlohmann::json;

bool ReadVector7(const json& value, std::array<double, 7>* out) {
  if (!value.is_array() || value.size() != 7) {
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    if (!value[i].is_number()) {
      return false;
    }
    const double v = value[i].get<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    (*out)[i] = v;
  }
  return true;
}

bool ParseActionSpace(const json& root, ActionSpace* out) {
  const std::string action_space = root.value("action_space", std::string("joint_position_absolute"));
  if (action_space == "joint_position_absolute") {
    *out = ActionSpace::kJointPositionAbsolute;
    return true;
  }
  return false;
}

bool ParsePolicyAction(const std::string& payload,
                       uint64_t receive_ns,
                       uint64_t fallback_sequence_id,
                       PolicyActionCommand* out) {
  const json root = json::parse(payload, nullptr, false);
  if (root.is_discarded() || !root.is_object()) {
    return false;
  }

  PolicyActionCommand cmd{};
  cmd.timestamp_ns = root.value("timestamp_ns", receive_ns);
  cmd.sequence_id = root.value("sequence_id", fallback_sequence_id);
  cmd.operator_request_id = root.value("operator_request_id", uint64_t{0});
  cmd.enabled = root.value("enabled", true);
  cmd.episode_start = root.value("episode_start", false);
  cmd.episode_end = root.value("episode_end", false);
  cmd.request_rehome = root.value("request_rehome", false);
  if (!ParseActionSpace(root, &cmd.action.action_space)) {
    return false;
  }

  if (root.contains("joint_positions_rad")) {
    if (!ReadVector7(root["joint_positions_rad"], &cmd.action.joint_positions_rad)) {
      return false;
    }
  } else if (root.contains("action")) {
    if (!ReadVector7(root["action"], &cmd.action.joint_positions_rad)) {
      return false;
    }
  } else {
    return false;
  }
  if (!root.contains("gripper_command") || !root["gripper_command"].is_number()) {
    return false;
  }
  cmd.action.gripper_command = root["gripper_command"].get<double>();

  for (double v : cmd.action.delta_translation_m) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  for (double v : cmd.action.delta_rotation_rad) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  for (double v : cmd.action.joint_positions_rad) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  if (!std::isfinite(cmd.action.gripper_command)) {
    return false;
  }
  cmd.action.gripper_command = Clamp01(cmd.action.gripper_command);

  *out = cmd;
  return true;
}

}  // namespace

PolicyActionSource::PolicyActionSource(std::string bind_ip,
                                       uint16_t action_port,
                                       LatestPolicyActionBuffer* action_buffer,
                                       std::atomic<bool>* stop_requested)
    : bind_ip_(std::move(bind_ip)),
      action_port_(action_port),
      action_buffer_(action_buffer),
      stop_requested_(stop_requested) {}

PolicyActionSource::~PolicyActionSource() {
  Stop();
}

bool PolicyActionSource::Start() {
  if (running_.exchange(true, std::memory_order_acq_rel)) {
    return true;
  }

  sock_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock_ < 0) {
    running_.store(false, std::memory_order_release);
    std::cerr << "PolicyActionSource: socket() failed: " << std::strerror(errno) << "\n";
    return false;
  }

  int reuse = 1;
  (void)setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  timeval timeout{};
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;
  (void)setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(action_port_);
  if (inet_pton(AF_INET, bind_ip_.c_str(), &addr.sin_addr) != 1) {
    close(sock_);
    sock_ = -1;
    running_.store(false, std::memory_order_release);
    std::cerr << "PolicyActionSource: invalid bind IP " << bind_ip_ << "\n";
    return false;
  }
  if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(sock_);
    sock_ = -1;
    running_.store(false, std::memory_order_release);
    std::cerr << "PolicyActionSource: bind(" << bind_ip_ << ":" << action_port_
              << ") failed: " << std::strerror(errno) << "\n";
    return false;
  }

  thread_ = std::thread(&PolicyActionSource::Run, this);
  return true;
}

void PolicyActionSource::Stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  if (sock_ >= 0) {
    close(sock_);
    sock_ = -1;
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void PolicyActionSource::Run() {
  std::array<char, 4096> buffer{};
  while (running_.load(std::memory_order_acquire) &&
         !stop_requested_->load(std::memory_order_acquire)) {
    sockaddr_in src{};
    socklen_t src_len = sizeof(src);
    const ssize_t n = recvfrom(sock_,
                               buffer.data(),
                               buffer.size() - 1,
                               0,
                               reinterpret_cast<sockaddr*>(&src),
                               &src_len);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR || sock_ < 0) {
        continue;
      }
      dropped_count_.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }
    buffer[static_cast<size_t>(n)] = '\0';

    const uint64_t receive_ns = MonotonicNowNs();
    const uint64_t next_sequence_id = sequence_id_.fetch_add(1, std::memory_order_acq_rel) + 1;
    PolicyActionCommand cmd{};
    if (!ParsePolicyAction(std::string(buffer.data(), static_cast<size_t>(n)),
                           receive_ns,
                           next_sequence_id,
                           &cmd)) {
      dropped_count_.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }

    action_buffer_->Publish(cmd);
    received_count_.fetch_add(1, std::memory_order_acq_rel);
    last_packet_time_ns_.store(receive_ns, std::memory_order_release);
  }
}

}  // namespace teleop
