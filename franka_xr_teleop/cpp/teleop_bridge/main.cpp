#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <franka/robot.h>

#include "common_types.h"
#include "config_loader.h"
#include "franka_controller.h"
#include "observation_pub.h"
#include "policy_action_source.h"
#include "xrobotics_source.h"

namespace {

std::atomic<bool> g_stop_requested{false};

void HandleSignal(int) {
  g_stop_requested.store(true, std::memory_order_release);
}

struct Options {
  std::string config_dir = "configs";
  bool dry_run_override = false;
  bool dry_run = false;
  bool allow_motion_override = false;
  bool allow_motion = true;
  bool robot_ip_override = false;
  std::string robot_ip;
  bool obs_ip_override = false;
  std::string obs_ip;
  bool obs_port_override = false;
  uint16_t obs_port = 28081;
  bool control_mode_override = false;
  teleop::ControlMode control_mode = teleop::ControlMode::kPose;
  bool control_source_override = false;
  teleop::ControlSource control_source = teleop::ControlSource::kXr;
  bool policy_bind_ip_override = false;
  std::string policy_bind_ip;
  bool policy_action_port_override = false;
  uint16_t policy_action_port = 28082;
  bool trace_enabled = false;
  std::string trace_dir = "teleop_trace";
  uint32_t trace_planner_decimation = 1;
  uint32_t trace_rt_decimation = 1;
  bool save_home = false;
};

void PrintUsage(const char* prog) {
  std::cout << "Usage:\n"
            << "  " << prog << " [--config-dir configs] [--dry-run] [--no-motion]\n"
            << "             [--robot-ip <ip>] [--obs-ip <ip>] [--obs-port <port>]\n"
            << "             [--control-mode <pose|position>] [--control-source <xr|policy>]\n"
            << "             [--policy-bind-ip <ip>] [--policy-action-port <port>] [--save-home]\n"
            << "             [--trace-dir <dir>] [--trace-planner-decimation <N>] "
               "[--trace-rt-decimation <N>]\n\n"
            << "Examples:\n"
            << "  " << prog << " --dry-run\n"
            << "  " << prog << " --robot-ip 192.168.2.200 --control-mode position\n"
            << "  " << prog << " --robot-ip 192.168.2.200 --control-source policy\n"
            << "  " << prog << " --robot-ip 192.168.2.200 --save-home\n"
            << "  " << prog << " --robot-ip 192.168.2.200 --trace-dir trace_run_01\n";
}

bool ParseArgs(int argc, char** argv, Options* out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--config-dir") {
      if (i + 1 >= argc) {
        return false;
      }
      out->config_dir = argv[++i];
      continue;
    }
    if (arg == "--dry-run") {
      out->dry_run_override = true;
      out->dry_run = true;
      continue;
    }
    if (arg == "--no-motion") {
      out->allow_motion_override = true;
      out->allow_motion = false;
      continue;
    }
    if (arg == "--robot-ip") {
      if (i + 1 >= argc) {
        return false;
      }
      out->robot_ip_override = true;
      out->robot_ip = argv[++i];
      continue;
    }
    if (arg == "--obs-ip") {
      if (i + 1 >= argc) {
        return false;
      }
      out->obs_ip_override = true;
      out->obs_ip = argv[++i];
      continue;
    }
    if (arg == "--obs-port") {
      if (i + 1 >= argc) {
        return false;
      }
      out->obs_port_override = true;
      out->obs_port = static_cast<uint16_t>(std::stoi(argv[++i]));
      continue;
    }
    if (arg == "--control-mode") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!teleop::ParseControlMode(argv[++i], &out->control_mode) ||
          out->control_mode == teleop::ControlMode::kHold) {
        return false;
      }
      out->control_mode_override = true;
      continue;
    }
    if (arg == "--control-source") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!teleop::ParseControlSource(argv[++i], &out->control_source)) {
        return false;
      }
      out->control_source_override = true;
      continue;
    }
    if (arg == "--policy-bind-ip") {
      if (i + 1 >= argc) {
        return false;
      }
      out->policy_bind_ip_override = true;
      out->policy_bind_ip = argv[++i];
      continue;
    }
    if (arg == "--policy-action-port") {
      if (i + 1 >= argc) {
        return false;
      }
      out->policy_action_port_override = true;
      out->policy_action_port = static_cast<uint16_t>(std::stoi(argv[++i]));
      continue;
    }
    if (arg == "--trace-dir") {
      if (i + 1 >= argc) {
        return false;
      }
      out->trace_enabled = true;
      out->trace_dir = argv[++i];
      continue;
    }
    if (arg == "--trace-planner-decimation") {
      if (i + 1 >= argc) {
        return false;
      }
      out->trace_enabled = true;
      out->trace_planner_decimation = static_cast<uint32_t>(std::stoul(argv[++i]));
      if (out->trace_planner_decimation == 0) {
        return false;
      }
      continue;
    }
    if (arg == "--trace-rt-decimation") {
      if (i + 1 >= argc) {
        return false;
      }
      out->trace_enabled = true;
      out->trace_rt_decimation = static_cast<uint32_t>(std::stoul(argv[++i]));
      if (out->trace_rt_decimation == 0) {
        return false;
      }
      continue;
    }
    if (arg == "--save-home") {
      out->save_home = true;
      continue;
    }
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
    return false;
  }
  return true;
}

uint64_t MonotonicNowNs() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::string FormatJointPositions(const std::array<double, 7>& q) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(8) << "[";
  for (size_t i = 0; i < q.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << q[i];
  }
  out << "]";
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  teleop::AppConfig config;
  std::string config_error;
  std::string resolved_config_dir = options.config_dir;
  if (!teleop::LoadAppConfig(options.config_dir, &config, &config_error)) {
    if (options.config_dir == "configs") {
      const std::string fallback_config_dir = "franka_xr_teleop/configs";
      if (!teleop::LoadAppConfig(fallback_config_dir, &config, &config_error)) {
        std::cerr << "Config error: " << config_error << "\n";
        return 1;
      }
      resolved_config_dir = fallback_config_dir;
    } else {
      std::cerr << "Config error: " << config_error << "\n";
      return 1;
    }
  }

  if (options.dry_run_override) {
    config.dry_run = options.dry_run;
  }
  if (options.allow_motion_override) {
    config.bridge.allow_motion = options.allow_motion;
  }
  if (options.robot_ip_override) {
    config.bridge.robot_ip = options.robot_ip;
  }
  if (options.obs_ip_override) {
    config.observation_ip = options.obs_ip;
  }
  if (options.obs_port_override) {
    config.observation_port = options.obs_port;
  }
  if (options.control_mode_override) {
    config.bridge.teleop.control_mode = options.control_mode;
  }
  if (options.control_source_override) {
    config.bridge.control_source = options.control_source;
  }
  if (options.policy_bind_ip_override) {
    config.bridge.policy.bind_ip = options.policy_bind_ip;
  }
  if (options.policy_action_port_override) {
    config.bridge.policy.action_port = options.policy_action_port;
  }

  if (options.save_home) {
    try {
      franka::Robot robot(config.bridge.robot_ip);
      const franka::RobotState state = robot.readOnce();
      if (!teleop::SaveStartJointPositions(
              resolved_config_dir, state.q, &config_error)) {
        std::cerr << "Failed to save home configuration: " << config_error << "\n";
        return 3;
      }

      std::cout << "Saved teleop.start_joint_positions_rad to " << resolved_config_dir
                << "/teleop.yaml = " << FormatJointPositions(state.q) << "\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "Failed to read current robot joints from " << config.bridge.robot_ip << ": "
                << e.what() << "\n";
      return 3;
    }
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  teleop::LatestCommandBuffer command_buffer;
  teleop::LatestPolicyActionBuffer policy_action_buffer;
  teleop::LatestObservationBuffer observation_buffer;

  std::unique_ptr<teleop::XrRoboticsSource> xr_source;
  std::unique_ptr<teleop::PolicyActionSource> policy_action_source;
  if (config.bridge.control_source == teleop::ControlSource::kXr) {
    xr_source = std::make_unique<teleop::XrRoboticsSource>(&command_buffer, &g_stop_requested);
    if (!xr_source->Start()) {
      std::cerr << "Failed to initialize XRoboToolkit SDK source.\n"
                << "Ensure XRoboToolkit PC Service is installed and running "
                << "(for example /opt/apps/roboticsservice/runService.sh).\n";
      return 2;
    }
  } else {
    policy_action_source = std::make_unique<teleop::PolicyActionSource>(
        config.bridge.policy.bind_ip,
        config.bridge.policy.action_port,
        &policy_action_buffer,
        &g_stop_requested);
    if (!policy_action_source->Start()) {
      std::cerr << "Failed to initialize policy action source on "
                << config.bridge.policy.bind_ip << ":" << config.bridge.policy.action_port << ".\n";
      return 2;
    }
    std::cout << "Policy action source listening on udp://" << config.bridge.policy.bind_ip << ":"
              << config.bridge.policy.action_port
              << " timeout_s=" << config.bridge.policy.command_timeout_s << "\n";
  }

  teleop::ObservationPublisher observation_pub(config.observation_ip, config.observation_port);
  observation_pub.Start();

  std::thread observation_thread([&]() {
    while (!g_stop_requested.load(std::memory_order_acquire)) {
      const teleop::RobotObservation obs = observation_buffer.ReadLatest();
      observation_pub.Publish(obs);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  if (config.dry_run) {
    std::cout << "Dry-run mode: control_source=" << teleop::ToString(config.bridge.control_source)
              << "\n";
    uint64_t last_print_ns = 0;
    while (!g_stop_requested.load(std::memory_order_acquire)) {
      const uint64_t now_ns = MonotonicNowNs();
      if (now_ns - last_print_ns > 500000000ULL) {
        if (config.bridge.control_source == teleop::ControlSource::kXr) {
          const teleop::XRCommand cmd = command_buffer.ReadLatest();
          const uint64_t age_ns = now_ns > cmd.timestamp_ns ? (now_ns - cmd.timestamp_ns) : 0;
          std::cout << "server_connected=" << (xr_source->server_connected() ? 1 : 0)
                    << " device_connected=" << (xr_source->device_connected() ? 1 : 0)
                    << " rx_count=" << xr_source->received_count()
                    << " dropped=" << xr_source->dropped_count()
                    << " seq=" << cmd.sequence_id
                    << " age_ms=" << (age_ns * 1e-6)
                    << " right_grip=" << cmd.control_trigger_value
                    << " right_trigger=" << cmd.gripper_trigger_value
                    << " A=" << (cmd.button_a ? 1 : 0)
                    << " B=" << (cmd.button_b ? 1 : 0)
                    << " right_axis_click=" << (cmd.right_axis_click ? 1 : 0)
                    << "\n";
        } else {
          const teleop::PolicyActionCommand cmd = policy_action_buffer.ReadLatest();
          const uint64_t age_ns = now_ns > cmd.timestamp_ns ? (now_ns - cmd.timestamp_ns) : 0;
          std::cout << "policy_rx_count=" << policy_action_source->received_count()
                    << " dropped=" << policy_action_source->dropped_count()
                    << " seq=" << cmd.sequence_id
                    << " age_ms=" << (age_ns * 1e-6)
                    << " enabled=" << (cmd.enabled ? 1 : 0)
                    << " op_request_id=" << cmd.operator_request_id
                    << " request_rehome=" << (cmd.request_rehome ? 1 : 0)
                    << " action_space=" << teleop::ToString(cmd.action.action_space);
          if (cmd.action.action_space == teleop::ActionSpace::kJointPositionAbsolute) {
            std::cout << " joint_positions=[" << cmd.action.joint_positions_rad[0] << ","
                      << cmd.action.joint_positions_rad[1] << ","
                      << cmd.action.joint_positions_rad[2] << ","
                      << cmd.action.joint_positions_rad[3] << ","
                      << cmd.action.joint_positions_rad[4] << ","
                      << cmd.action.joint_positions_rad[5] << ","
                      << cmd.action.joint_positions_rad[6] << "]";
          } else {
            std::cout << " action=[" << cmd.action.delta_translation_m[0] << ","
                      << cmd.action.delta_translation_m[1] << ","
                      << cmd.action.delta_translation_m[2] << ","
                      << cmd.action.delta_rotation_rad[0] << ","
                      << cmd.action.delta_rotation_rad[1] << ","
                      << cmd.action.delta_rotation_rad[2] << "]";
          }
          std::cout << " gripper=" << cmd.action.gripper_command << "\n";
        }
        last_print_ns = now_ns;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } else {
    teleop::FrankaControllerOptions controller_options;
    controller_options.robot_ip = config.bridge.robot_ip;
    controller_options.trace.enabled = options.trace_enabled;
    controller_options.trace.output_dir = options.trace_dir;
    controller_options.trace.planner_decimation = options.trace_planner_decimation;
    controller_options.trace.rt_decimation = options.trace_rt_decimation;
    teleop::FrankaTeleopController controller(
        controller_options,
        config.bridge,
        &command_buffer,
        &policy_action_buffer,
        &observation_buffer);

    const int rc = controller.Run(&g_stop_requested);
    g_stop_requested.store(true, std::memory_order_release);
    if (xr_source) {
      xr_source->Stop();
    }
    if (policy_action_source) {
      policy_action_source->Stop();
    }
    if (observation_thread.joinable()) {
      observation_thread.join();
    }
    observation_pub.Stop();
    return rc;
  }

  g_stop_requested.store(true, std::memory_order_release);
  if (xr_source) {
    xr_source->Stop();
  }
  if (policy_action_source) {
    policy_action_source->Stop();
  }
  if (observation_thread.joinable()) {
    observation_thread.join();
  }
  observation_pub.Stop();
  return 0;
}
