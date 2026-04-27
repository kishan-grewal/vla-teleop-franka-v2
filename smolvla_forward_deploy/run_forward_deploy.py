from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch


# Franka Emika Panda hard joint limits (rad).
FRANKA_Q_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
FRANKA_Q_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_observation_timestamp_ns(obs_packet: dict[str, Any]) -> int | None:
    raw_ts = obs_packet.get("timestamp_ns")
    if raw_ts is None:
        return None
    try:
        ts = int(raw_ts)
    except (TypeError, ValueError):
        return None
    return ts if ts >= 0 else None


def parse_observation_sequence_id(obs_packet: dict[str, Any]) -> int | None:
    raw_seq = obs_packet.get("sequence_id")
    if raw_seq is None:
        return None
    try:
        seq = int(raw_seq)
    except (TypeError, ValueError):
        return None
    return seq if seq >= 0 else None


def status_fault_reasons(obs_packet: dict[str, Any], safety_cfg: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = obs_packet.get("status", {})
    if not isinstance(status, dict):
        return reasons

    fault_flags = status.get("fault_flags", {})
    if not isinstance(fault_flags, dict):
        fault_flags = {}

    blocked_faults = safety_cfg.get(
        "blocked_fault_flags",
        [
            "packet_timeout",
            "jump_rejected",
            "workspace_clamped",
            "robot_not_ready",
            "control_exception",
            "ik_rejected",
        ],
    )
    for key in blocked_faults:
        if bool(fault_flags.get(str(key), False)):
            reasons.append(f"fault_flag:{key}")

    if bool(safety_cfg.get("require_target_fresh", False)) and not bool(status.get("target_fresh", True)):
        reasons.append("target_not_fresh")

    if bool(safety_cfg.get("require_teleop_active", False)) and not bool(status.get("teleop_active", True)):
        reasons.append("teleop_not_active")

    required_control_modes = safety_cfg.get("allowed_control_modes", [])
    if required_control_modes:
        mode = str(status.get("control_mode", "")).upper()
        normalized = [str(v).upper() for v in required_control_modes]
        if mode not in normalized:
            reasons.append(f"control_mode_blocked:{mode}")

    return reasons


def require_finite(values: list[float], name: str) -> None:
    for idx, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value in {name}[{idx}]: {value}")


def build_hold_action(current_q: list[float], current_gripper: float) -> list[float]:
    return [float(v) for v in current_q] + [float(current_gripper)]


def smooth_joint_targets(
    desired_q: list[float],
    prev_q: list[float] | None,
    prev_dq: list[float],
    dt: float,
    safety_cfg: dict[str, Any],
) -> tuple[list[float], list[float]]:
    if len(desired_q) != 7:
        raise ValueError("desired_q must have 7 joints")
    if prev_q is None:
        return [float(v) for v in desired_q], [0.0] * 7

    max_vel = float(safety_cfg.get("max_joint_speed_rad_s", 0.35))
    max_acc = float(safety_cfg.get("max_joint_acceleration_rad_s2", 1.5))
    max_step_cfg = float(safety_cfg.get("max_joint_step_rad", 0.008))
    alpha = float(safety_cfg.get("target_smoothing_alpha", 0.25))

    safe_dt = max(1e-6, dt)
    max_step_from_vel = max_vel * safe_dt
    max_step = min(max_step_cfg, max_step_from_vel)
    max_dv = max_acc * safe_dt
    alpha = clamp(alpha, 0.0, 1.0)

    out_q: list[float] = []
    out_dq: list[float] = []
    for i in range(7):
        raw_vel = (float(desired_q[i]) - float(prev_q[i])) / safe_dt
        blended_vel = alpha * raw_vel + (1.0 - alpha) * float(prev_dq[i])
        vel_limited = clamp(blended_vel, -max_vel, max_vel)
        accel_limited = float(prev_dq[i]) + clamp(vel_limited - float(prev_dq[i]), -max_dv, max_dv)
        step = clamp(accel_limited * safe_dt, -max_step, max_step)
        q_i = float(prev_q[i]) + step
        out_q.append(q_i)
        out_dq.append(step / safe_dt)

    return out_q, out_dq


def apply_franka_safety(
    raw_action: list[float],
    current_q: list[float],
    current_gripper: float,
    elapsed_s: float,
    safety_cfg: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    if len(raw_action) < 8:
        raise ValueError("Model action must contain at least 8 dims for Franka")
    if len(current_q) != 7:
        raise ValueError("Current Franka q must contain 7 joints")

    margin = float(safety_cfg.get("joint_position_margin_rad", 0.10))
    max_step_rad = float(safety_cfg.get("max_joint_step_rad", 0.02))
    max_speed_rad_s = float(safety_cfg.get("max_joint_speed_rad_s", 0.35))
    gripper_min = float(safety_cfg.get("gripper_width_min_m", 0.0))
    gripper_max = float(safety_cfg.get("gripper_width_max_m", 0.08))

    step_limit = min(max_step_rad, max(0.0, elapsed_s) * max_speed_rad_s)
    if step_limit <= 0.0:
        step_limit = max_step_rad

    clipped_indices: list[int] = []
    safe_joints: list[float] = []
    for i in range(7):
        low = FRANKA_Q_MIN[i] + margin
        high = FRANKA_Q_MAX[i] - margin
        target = clamp(float(raw_action[i]), low, high)
        delta = target - float(current_q[i])
        if abs(delta) > step_limit:
            target = float(current_q[i]) + math.copysign(step_limit, delta)
            clipped_indices.append(i)
        safe_joints.append(target)

    safe_gripper = clamp(float(raw_action[7]), gripper_min, gripper_max)
    if safe_gripper != float(raw_action[7]):
        clipped_indices.append(7)

    safe_action = safe_joints + [safe_gripper]
    return safe_action, {
        "mode": "active",
        "clipped_indices": clipped_indices,
        "max_joint_step_rad_applied": step_limit,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_lerobot_importable(lerobot_root: Path) -> None:
    src = lerobot_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"Could not find lerobot source at {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def pick_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config_device)


def make_image(camera_cfg: dict[str, Any], device: torch.device, step: int) -> torch.Tensor:
    mode = camera_cfg.get("mode", "dummy")
    h = int(camera_cfg.get("height", 480))
    w = int(camera_cfg.get("width", 640))

    if mode == "dummy":
        # Keep this deterministic and non-zero so processors do not see a blank frame.
        value = ((step % 255) + 1) / 255.0
        return torch.full((1, 3, h, w), value, dtype=torch.float32, device=device)

    if mode == "webcam":
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("camera.mode=webcam requires opencv-python") from exc

        index = int(camera_cfg.get("index", 0))
        cap = cv2.VideoCapture(index)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read webcam frame from index={index}")

        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(device)

    raise ValueError(f"Unsupported camera mode: {mode}")


def extract_state(obs_packet: dict[str, Any], state_key: str, device: torch.device) -> torch.Tensor:
    state = obs_packet.get(state_key, {})
    q = state.get("q", [])
    gripper = state.get("gripper_width", 0.0)

    if len(q) != 7:
        raise ValueError("Observation packet missing robot_state.q with 7 joints")

    values = [float(v) for v in q] + [float(gripper)]
    require_finite(values, "observation.state")
    return torch.tensor([values], dtype=torch.float32, device=device)


def build_action_packet(
    action: list[float],
    task: str,
    sequence_id: int,
    safety: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_ns": time.time_ns(),
        "sequence_id": sequence_id,
        "source": "smolvla_forward_deploy",
        "task": task,
        "joint_targets_q1_q7": action[:7],
        "gripper_width": action[7],
        "raw_action": action,
        "safety": safety,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward deploy SmolVLA for Franka from UDP observations")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "deploy_config.example.json",
        help="Path to deployment config JSON",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    config_dir = args.config.resolve().parent

    lerobot_root = Path(cfg["lerobot_root"])
    if not lerobot_root.is_absolute():
        lerobot_root = (config_dir / lerobot_root).resolve()

    model_path = Path(cfg["model_path"])
    if not model_path.is_absolute():
        model_path = (config_dir / model_path).resolve()

    ensure_lerobot_importable(lerobot_root)

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    device = pick_device(str(cfg.get("device", "auto")))
    task = str(cfg["task"])
    action_dim = int(cfg.get("action_dim", 8))
    loop_hz = float(cfg.get("loop_hz", 10.0))
    max_steps = int(cfg.get("max_steps", 0))
    print_every = int(cfg.get("print_every", 10))

    if action_dim < 8:
        raise ValueError("action_dim must be >= 8 for Franka (7 joints + gripper)")

    safety_cfg = dict(cfg.get("safety", {}))
    safety_enabled = bool(safety_cfg.get("enabled", True))
    require_armed_file = bool(safety_cfg.get("require_armed_file", True))
    armed_file = Path(safety_cfg.get("armed_file", "./ARMED"))
    if not armed_file.is_absolute():
        armed_file = (config_dir / armed_file).resolve()

    estop_file = Path(safety_cfg.get("estop_file", "./ESTOP"))
    if not estop_file.is_absolute():
        estop_file = (config_dir / estop_file).resolve()

    max_observation_age_ms = float(safety_cfg.get("max_observation_age_ms", 150.0))
    hold_on_policy_error = bool(safety_cfg.get("hold_on_policy_error", True))
    socket_timeout_ms = int(safety_cfg.get("socket_timeout_ms", 500))
    hold_on_stale_observation = bool(safety_cfg.get("hold_on_stale_observation", True))
    fail_closed_on_missing_timestamp = bool(safety_cfg.get("fail_closed_on_missing_timestamp", True))

    print(f"Loading SmolVLA policy from: {model_path}")
    policy = SmolVLAPolicy.from_pretrained(str(model_path)).to(device)
    policy.eval()
    policy.reset()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(model_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    obs_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    obs_sock.bind((str(cfg.get("observation_udp_host", "127.0.0.1")), int(cfg.get("observation_udp_port", 28081))))
    obs_sock.settimeout(max(0.0, socket_timeout_ms / 1000.0))

    action_addr = (str(cfg.get("action_udp_host", "127.0.0.1")), int(cfg.get("action_udp_port", 29001)))
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    save_actions_path = cfg.get("save_actions_jsonl")
    actions_file = None
    if save_actions_path:
        output_path = Path(save_actions_path)
        if not output_path.is_absolute():
            output_path = Path(args.config).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        actions_file = output_path.open("a", encoding="utf-8")

    camera_cfg = cfg.get("camera", {})
    image_key = str(camera_cfg.get("image_key", "observation.images.top"))
    state_key = str(cfg.get("state_key", "robot_state"))

    dt = 1.0 / max(loop_hz, 1e-6)
    next_step_time = time.perf_counter()
    steps = 0
    last_obs_state: tuple[list[float], float] | None = None
    last_action_time = time.perf_counter()
    last_obs_sequence_id: int | None = None
    prev_safe_q: list[float] | None = None
    prev_safe_dq: list[float] = [0.0] * 7

    print(
        f"Listening for observations on {cfg.get('observation_udp_host', '127.0.0.1')}:{cfg.get('observation_udp_port', 28081)}"
    )
    print(f"Publishing actions to {action_addr[0]}:{action_addr[1]}")
    print(f"Safety enabled: {safety_enabled}")
    if require_armed_file:
        print(f"Arming gate file required: {armed_file}")
    print(f"E-stop file path: {estop_file}")

    try:
        while True:
            try:
                raw, _ = obs_sock.recvfrom(65535)
                obs_packet = json.loads(raw.decode("utf-8"))
            except socket.timeout:
                if last_obs_state is None:
                    continue
                current_q, current_gripper = last_obs_state
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "socket_timeout",
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            state = obs_packet.get(state_key, {})
            current_q = [float(v) for v in state.get("q", [])]
            current_gripper = float(state.get("gripper_width", 0.0))
            if len(current_q) != 7:
                raise ValueError("Observation packet missing robot_state.q with 7 joints")
            require_finite(current_q + [current_gripper], "robot_state")
            last_obs_state = (current_q, current_gripper)

            obs_seq = parse_observation_sequence_id(obs_packet)
            if bool(safety_cfg.get("enforce_monotonic_sequence", True)) and obs_seq is not None:
                if last_obs_sequence_id is not None and obs_seq <= last_obs_sequence_id:
                    hold = build_hold_action(current_q, current_gripper)
                    packet = build_action_packet(
                        hold,
                        task,
                        steps,
                        {
                            "mode": "hold",
                            "reason": "non_monotonic_observation_sequence",
                            "clipped_indices": [],
                        },
                    )
                    action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                    prev_safe_q = None
                    prev_safe_dq = [0.0] * 7
                    steps += 1
                    continue
                last_obs_sequence_id = obs_seq

            status_reasons = status_fault_reasons(obs_packet, safety_cfg)
            if status_reasons:
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "status_fault_gate",
                        "status_reasons": status_reasons,
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                if steps % max(print_every, 1) == 0:
                    print(f"Status/fault gate active: {status_reasons}")
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            if estop_file.exists():
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "estop_file_present",
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                if steps % max(print_every, 1) == 0:
                    print("E-stop asserted; publishing hold action.")
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            if require_armed_file and not armed_file.exists():
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "not_armed",
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                if steps % max(print_every, 1) == 0:
                    print("Arming file missing; publishing hold action.")
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            obs_ts_ns = parse_observation_timestamp_ns(obs_packet)
            now_ns = time.time_ns()
            if obs_ts_ns is None:
                if fail_closed_on_missing_timestamp:
                    hold = build_hold_action(current_q, current_gripper)
                    packet = build_action_packet(
                        hold,
                        task,
                        steps,
                        {
                            "mode": "hold",
                            "reason": "missing_observation_timestamp",
                            "clipped_indices": [],
                        },
                    )
                    action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                    prev_safe_q = None
                    prev_safe_dq = [0.0] * 7
                    steps += 1
                    continue
                obs_age_ms = 0.0
            else:
                obs_age_ms = max(0.0, (now_ns - obs_ts_ns) / 1e6)

            if hold_on_stale_observation and obs_age_ms > max_observation_age_ms:
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "stale_observation",
                        "observation_age_ms": obs_age_ms,
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                if steps % max(print_every, 1) == 0:
                    print(f"Stale observation ({obs_age_ms:.1f} ms); publishing hold action.")
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            state_tensor = extract_state(obs_packet, state_key, device)
            image_tensor = make_image(camera_cfg, device, steps)

            raw_obs = {
                image_key: image_tensor,
                "observation.state": state_tensor,
                "task": task,
            }

            try:
                processed_obs = preprocess(raw_obs)
                action_tensor = policy.select_action(processed_obs)
                action_denorm = postprocess(action_tensor)
                franka_action = action_denorm[0, :action_dim].detach().cpu().tolist()
                require_finite([float(v) for v in franka_action[:8]], "policy_action")
            except Exception:
                if not hold_on_policy_error:
                    raise
                hold = build_hold_action(current_q, current_gripper)
                packet = build_action_packet(
                    hold,
                    task,
                    steps,
                    {
                        "mode": "hold",
                        "reason": "policy_error",
                        "clipped_indices": [],
                    },
                )
                action_sock.sendto(json.dumps(packet).encode("utf-8"), action_addr)
                if steps % max(print_every, 1) == 0:
                    print("Policy error; publishing hold action.")
                prev_safe_q = None
                prev_safe_dq = [0.0] * 7
                steps += 1
                continue

            elapsed_s = max(1e-3, time.perf_counter() - last_action_time)
            if safety_enabled:
                safe_action, safety_meta = apply_franka_safety(
                    franka_action,
                    current_q,
                    current_gripper,
                    elapsed_s,
                    safety_cfg,
                )
            else:
                safe_action = franka_action[:8]
                safety_meta = {"mode": "active", "clipped_indices": []}

            smooth_q, smooth_dq = smooth_joint_targets(
                safe_action[:7],
                prev_safe_q,
                prev_safe_dq,
                elapsed_s,
                safety_cfg,
            )
            prev_safe_q = smooth_q
            prev_safe_dq = smooth_dq
            safe_action = smooth_q + [safe_action[7]]
            safety_meta["smoothing_alpha"] = float(safety_cfg.get("target_smoothing_alpha", 0.25))

            safety_meta["observation_age_ms"] = obs_age_ms
            packet = build_action_packet(safe_action, task, steps, safety_meta)
            payload = json.dumps(packet).encode("utf-8")
            action_sock.sendto(payload, action_addr)
            last_action_time = time.perf_counter()

            if actions_file is not None:
                actions_file.write(json.dumps(packet) + "\n")
                actions_file.flush()

            if steps % max(print_every, 1) == 0:
                joints_str = ", ".join(f"{x:.3f}" for x in packet["joint_targets_q1_q7"])
                mode = packet.get("safety", {}).get("mode", "unknown")
                clipped = packet.get("safety", {}).get("clipped_indices", [])
                print(
                    f"step={steps} mode={mode} joints=[{joints_str}] "
                    f"gripper={packet['gripper_width']:.4f} clipped={clipped}"
                )

            steps += 1
            if max_steps > 0 and steps >= max_steps:
                break

            next_step_time += dt
            now = time.perf_counter()
            if next_step_time > now:
                time.sleep(next_step_time - now)
            else:
                next_step_time = now
    finally:
        obs_sock.close()
        action_sock.close()
        if actions_file is not None:
            actions_file.close()


if __name__ == "__main__":
    main()
