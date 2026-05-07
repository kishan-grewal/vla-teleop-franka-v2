#include "observation_pub.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <iomanip>
#include <sstream>

#include "teleop_state_machine.h"

namespace teleop {
namespace {

template <size_t N>
void AppendArray(std::ostringstream* os, const std::array<double, N>& values) {
  *os << '[';
  for (size_t i = 0; i < N; ++i) {
    *os << values[i];
    if (i + 1 < N) {
      *os << ',';
    }
  }
  *os << ']';
}

}  // namespace

ObservationPublisher::ObservationPublisher(std::string dst_ip, uint16_t dst_port)
    : dst_ip_(std::move(dst_ip)), dst_port_(dst_port) {}

ObservationPublisher::~ObservationPublisher() {
  Stop();
}

bool ObservationPublisher::Start() {
  if (started_) {
    return true;
  }
  sock_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock_ < 0) {
    return false;
  }
  started_ = true;
  return true;
}

void ObservationPublisher::Stop() {
  if (sock_ >= 0) {
    close(sock_);
    sock_ = -1;
  }
  started_ = false;
}

std::string ObservationPublisher::ToJson(const RobotObservation& obs) const {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(6);

  ss << '{';
  ss << "\"timestamp_ns\":" << obs.timestamp_ns << ',';
  ss << "\"robot_state\":{";
  ss << "\"q\":";
  AppendArray(&ss, obs.q);
  ss << ',';
  ss << "\"q_cmd\":";
  AppendArray(&ss, obs.q_cmd);
  ss << ',';
  ss << "\"dq\":";
  AppendArray(&ss, obs.dq);
  ss << ',';
  ss << "\"tcp_position_xyz\":";
  AppendArray(&ss, obs.tcp_pose.p);
  ss << ',';
  ss << "\"tcp_orientation_xyzw\":";
  AppendArray(&ss, obs.tcp_pose.q);
  ss << ',';
  ss << "\"gripper_width\":" << obs.gripper_width << ',';
  ss << "\"gripper_state\":\"" << ToString(obs.gripper_state) << "\"";
  ss << "},";

  ss << "\"desired_target_state\":{";
  ss << "\"tcp_position_xyz\":";
  AppendArray(&ss, obs.desired_target_tcp_pose.p);
  ss << ',';
  ss << "\"tcp_orientation_xyzw\":";
  AppendArray(&ss, obs.desired_target_tcp_pose.q);
  ss << "},";

  ss << "\"commanded_target_state\":{";
  ss << "\"tcp_position_xyz\":";
  AppendArray(&ss, obs.commanded_target_tcp_pose.p);
  ss << ',';
  ss << "\"tcp_orientation_xyzw\":";
  AppendArray(&ss, obs.commanded_target_tcp_pose.q);
  ss << "},";

  ss << "\"executed_action\":{";
  ss << "\"action_space\":\"" << ToString(obs.executed_action.action_space) << "\",";
  ss << "\"cartesian_delta_translation\":";
  AppendArray(&ss, obs.executed_action.delta_translation_m);
  ss << ',';
  ss << "\"cartesian_delta_rotation\":";
  AppendArray(&ss, obs.executed_action.delta_rotation_rad);
  ss << ',';
  ss << "\"joint_positions_rad\":";
  AppendArray(&ss, obs.executed_action.joint_positions_rad);
  ss << ',';
  ss << "\"gripper_command\":" << obs.executed_action.gripper_command;
  ss << "},";

  ss << "\"status\":{";
  ss << "\"control_mode\":\"" << ToString(obs.control_mode) << "\",";
  ss << "\"teleop_state\":\"" << ToString(obs.teleop_state) << "\",";
  ss << "\"packet_age_ns\":" << obs.packet_age_ns << ',';
  ss << "\"target_age_ns\":" << obs.target_age_ns << ',';
  ss << "\"target_fresh\":" << (obs.target_fresh ? "true" : "false") << ',';
  ss << "\"teleop_active\":" << (obs.teleop_active ? "true" : "false") << ',';
  ss << "\"episode_start\":" << (obs.episode_start ? "true" : "false") << ',';
  ss << "\"episode_end\":" << (obs.episode_end ? "true" : "false") << ',';
  ss << "\"target_manipulability\":" << obs.target_manipulability << ',';
  ss << "\"fault_flags\":{";
  ss << "\"packet_timeout\":" << (obs.faults.packet_timeout ? "true" : "false") << ',';
  ss << "\"jump_rejected\":" << (obs.faults.jump_rejected ? "true" : "false") << ',';
  ss << "\"workspace_clamped\":" << (obs.faults.workspace_clamped ? "true" : "false") << ',';
  ss << "\"robot_not_ready\":" << (obs.faults.robot_not_ready ? "true" : "false") << ',';
  ss << "\"control_exception\":" << (obs.faults.control_exception ? "true" : "false") << ',';
  ss << "\"ik_rejected\":" << (obs.faults.ik_rejected ? "true" : "false") << ',';
  ss << "\"gripper_fault\":" << (obs.faults.gripper_fault ? "true" : "false");
  ss << "}}";
  ss << '}';

  return ss.str();
}

bool ObservationPublisher::Publish(const RobotObservation& obs) {
  if (!started_ || sock_ < 0) {
    return false;
  }

  sockaddr_in dst{};
  dst.sin_family = AF_INET;
  dst.sin_port = htons(dst_port_);
  if (inet_pton(AF_INET, dst_ip_.c_str(), &dst.sin_addr) != 1) {
    return false;
  }

  const std::string payload = ToJson(obs);
  const ssize_t sent = sendto(sock_,
                              payload.data(),
                              payload.size(),
                              0,
                              reinterpret_cast<sockaddr*>(&dst),
                              sizeof(dst));
  return sent == static_cast<ssize_t>(payload.size());
}

}  // namespace teleop
