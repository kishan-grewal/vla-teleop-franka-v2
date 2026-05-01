#!/usr/bin/env python3
"""Live UDP observation monitor for franka_xr_teleop."""

import argparse
import json
import math
import socket
import time
from typing import Any, Dict, List, Optional, Tuple


def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def quat_xyzw_to_rpy_deg(q: List[float]) -> List[float]:
    if len(q) != 4:
        return [0.0, 0.0, 0.0]
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def quat_xyzw_to_matrix(q: List[float]) -> List[List[float]]:
    if len(q) != 4:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def import_plot_dependencies() -> Tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("3D plotting requires matplotlib and numpy") from exc
    return plt, np


def init_plot(plot_title: str) -> Tuple[Any, Any, Any]:
    plt, np = import_plot_dependencies()
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    try:
        ax = fig.add_subplot(111, projection="3d")
    except ValueError as exc:
        raise RuntimeError(
            "matplotlib is installed, but 3D plotting is unavailable. "
            "This usually means mpl_toolkits.mplot3d is missing."
        ) from exc
    fig.canvas.manager.set_window_title(plot_title)
    return plt, fig, ax


def draw_frame(ax: Any, origin: List[float], rot: List[List[float]], axis_len: float, alpha: float, prefix: str) -> None:
    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["x", "y", "z"]
    for axis_idx in range(3):
        direction = [rot[0][axis_idx], rot[1][axis_idx], rot[2][axis_idx]]
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=colors[axis_idx],
            length=axis_len,
            normalize=True,
            arrow_length_ratio=0.15,
            alpha=alpha,
            linewidth=2.5 if alpha >= 0.99 else 1.6,
        )
        tip = [
            origin[0] + direction[0] * axis_len * 1.1,
            origin[1] + direction[1] * axis_len * 1.1,
            origin[2] + direction[2] * axis_len * 1.1,
        ]
        ax.text(tip[0], tip[1], tip[2], f"{prefix}-{labels[axis_idx]}", color=colors[axis_idx], fontsize=9)


def update_plot(
    ax: Any,
    desired_target_p: List[float],
    desired_target_q: List[float],
    commanded_target_p: List[float],
    commanded_target_q: List[float],
    desired_target_rpy: List[float],
    commanded_target_rpy: List[float],
) -> None:
    ax.cla()
    axis_len = 0.06
    desired_target_origin = [float(v) for v in desired_target_p[:3]]
    commanded_target_origin = [float(v) for v in commanded_target_p[:3]]
    desired_target_rot = quat_xyzw_to_matrix(desired_target_q)
    commanded_target_rot = quat_xyzw_to_matrix(commanded_target_q)

    draw_frame(ax, desired_target_origin, desired_target_rot, axis_len, 0.7, "des")
    draw_frame(ax, commanded_target_origin, commanded_target_rot, axis_len, 0.95, "cmd")

    ax.scatter(*desired_target_origin, color="tab:orange", s=35)
    ax.scatter(*commanded_target_origin, color="tab:purple", s=35)
    ax.text(
        desired_target_origin[0],
        desired_target_origin[1] + 0.03,
        desired_target_origin[2] - 0.03,
        "Desired TCP",
        fontsize=10,
    )
    ax.text(
        commanded_target_origin[0],
        commanded_target_origin[1] - 0.03,
        commanded_target_origin[2] - 0.03,
        "Commanded TCP",
        fontsize=10,
    )

    ax.set_title(
        "Live Desired vs Commanded TCP Pose\n"
        f"des xyz=({desired_target_origin[0]:+.3f}, {desired_target_origin[1]:+.3f}, {desired_target_origin[2]:+.3f}) m    "
        f"cmd xyz=({commanded_target_origin[0]:+.3f}, {commanded_target_origin[1]:+.3f}, {commanded_target_origin[2]:+.3f}) m\n"
        f"des rpy=({desired_target_rpy[0]:+.1f}, {desired_target_rpy[1]:+.1f}, {desired_target_rpy[2]:+.1f}) deg    "
        f"cmd rpy=({commanded_target_rpy[0]:+.1f}, {commanded_target_rpy[1]:+.1f}, {commanded_target_rpy[2]:+.1f}) deg"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    pad = 0.12
    min_x = min(desired_target_origin[0], commanded_target_origin[0]) - pad
    max_x = max(desired_target_origin[0], commanded_target_origin[0]) + pad
    min_y = min(desired_target_origin[1], commanded_target_origin[1]) - pad
    max_y = max(desired_target_origin[1], commanded_target_origin[1]) + pad
    min_z = min(desired_target_origin[2], commanded_target_origin[2]) - pad
    max_z = max(desired_target_origin[2], commanded_target_origin[2]) + pad
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(
            (
                max(max_x - min_x, 1e-3),
                max(max_y - min_y, 1e-3),
                max(max_z - min_z, 1e-3),
            )
        )
    ax.grid(True, alpha=0.3)


def format_faults(fault_flags: Dict[str, Any]) -> str:
    active = [k for k, v in fault_flags.items() if bool(v)]
    if not active:
        return "none"
    return ",".join(active)


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Listen to franka_xr_teleop UDP observations and print live debug stats. "
            "Run this while franka_xr_teleop_bridge is active."
        )
    )
    p.add_argument("--bind-ip", default="0.0.0.0", help="IP to bind for UDP receive (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=28081, help="UDP port to listen on (default: 28081)")
    p.add_argument(
        "--print-hz",
        type=float,
        default=10.0,
        help="How often to print summaries (default: 10 Hz)",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=0.5,
        help="Socket timeout in seconds (default: 0.5)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print full decoded JSON payload each packet instead of compact summary",
    )
    p.add_argument(
        "--plot-3d",
        action="store_true",
        help="Open a live 3D view of desired and commanded TCP poses in the robot base frame.",
    )
    p.add_argument(
        "--plot-hz",
        type=float,
        default=20.0,
        help="3D plot update rate when --plot-3d is enabled (default: 20 Hz).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_period_s = 1.0 / max(args.print_hz, 0.1)
    plot_period_s = 1.0 / max(args.plot_hz, 0.1)
    plt: Optional[Any] = None
    fig: Optional[Any] = None
    ax: Optional[Any] = None
    last_plot_t = 0.0

    if args.plot_3d:
        try:
            plt, fig, ax = init_plot("franka_xr_teleop live orientation")
        except RuntimeError as exc:
            print(str(exc))
            return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(max(args.timeout_s, 0.01))
    sock.bind((args.bind_ip, args.port))

    print(
        f"Listening on udp://{args.bind_ip}:{args.port} "
        f"(print_hz={args.print_hz:.2f}, raw={args.raw}, plot_3d={args.plot_3d})"
    )

    last_print_t = 0.0
    last_rate_t = time.monotonic()
    packets_since_rate = 0
    rx_hz = 0.0
    last_payload_t = 0.0

    while True:
        try:
            payload, addr = sock.recvfrom(65535)
            now = time.monotonic()
            packets_since_rate += 1
            last_payload_t = now

            elapsed = now - last_rate_t
            if elapsed >= 1.0:
                rx_hz = packets_since_rate / elapsed
                packets_since_rate = 0
                last_rate_t = now

            try:
                obs = json.loads(payload.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as e:
                if now - last_print_t >= print_period_s:
                    print(f"[json_error] from {addr[0]}:{addr[1]}: {e}")
                    last_print_t = now
                continue

            if args.raw:
                print(json.dumps(obs, separators=(",", ":"), sort_keys=True))
                continue

            if now - last_print_t < print_period_s:
                continue

            status = safe_get(obs, "status", default={}) or {}
            robot = safe_get(obs, "robot_state", default={}) or {}
            desired_target = safe_get(obs, "desired_target_state", default={}) or {}
            commanded_target = safe_get(obs, "commanded_target_state", default={}) or {}
            action = safe_get(obs, "executed_action", default={}) or {}
            faults = safe_get(status, "fault_flags", default={}) or {}

            teleop_state = safe_get(status, "teleop_state", default="N/A")
            control_mode = safe_get(status, "control_mode", default="N/A")
            teleop_active = bool(safe_get(status, "teleop_active", default=False))
            target_fresh = bool(safe_get(status, "target_fresh", default=False))
            episode_start = bool(safe_get(status, "episode_start", default=False))
            episode_end = bool(safe_get(status, "episode_end", default=False))
            packet_age_ms = float(safe_get(status, "packet_age_ns", default=0)) * 1e-6
            target_age_ms = float(safe_get(status, "target_age_ns", default=0)) * 1e-6
            manip = float(safe_get(status, "target_manipulability", default=0.0))

            tcp = safe_get(robot, "tcp_position_xyz", default=[0.0, 0.0, 0.0])
            tcp_q = safe_get(robot, "tcp_orientation_xyzw", default=[0.0, 0.0, 0.0, 1.0])
            desired_target_tcp = safe_get(desired_target, "tcp_position_xyz", default=[0.0, 0.0, 0.0])
            desired_target_tcp_q = safe_get(desired_target, "tcp_orientation_xyzw", default=[0.0, 0.0, 0.0, 1.0])
            commanded_target_tcp = safe_get(commanded_target, "tcp_position_xyz", default=[0.0, 0.0, 0.0])
            commanded_target_tcp_q = safe_get(
                commanded_target, "tcp_orientation_xyzw", default=[0.0, 0.0, 0.0, 1.0]
            )
            dq = safe_get(robot, "dq", default=[0.0] * 7)
            gripper_width = float(safe_get(robot, "gripper_width", default=0.0))
            gripper_state = safe_get(robot, "gripper_state", default="N/A")

            action_space = safe_get(action, "action_space", default="cartesian_delta")
            dpos = safe_get(action, "cartesian_delta_translation", default=[0.0, 0.0, 0.0])
            drot = safe_get(action, "cartesian_delta_rotation", default=[0.0, 0.0, 0.0])
            joint_positions = safe_get(action, "joint_positions_rad", default=[0.0] * 7)
            grip_cmd = float(safe_get(action, "gripper_command", default=0.0))

            dpos_n = vec_norm(dpos)
            drot_n = vec_norm(drot)
            joint_target_max = max(abs(float(x)) for x in joint_positions) if joint_positions else 0.0
            dq_max = max(abs(float(x)) for x in dq) if dq else 0.0
            stale_ms = (now - last_payload_t) * 1e3
            tcp_rpy = quat_xyzw_to_rpy_deg(tcp_q)
            desired_target_rpy = quat_xyzw_to_rpy_deg(desired_target_tcp_q)
            commanded_target_rpy = quat_xyzw_to_rpy_deg(commanded_target_tcp_q)
            desired_pos_err = [float(desired_target_tcp[i]) - float(tcp[i]) for i in range(3)]
            commanded_pos_err = [float(commanded_target_tcp[i]) - float(tcp[i]) for i in range(3)]
            desired_cmd_pos_delta = [
                float(commanded_target_tcp[i]) - float(desired_target_tcp[i]) for i in range(3)
            ]
            desired_rpy_err = [desired_target_rpy[i] - tcp_rpy[i] for i in range(3)]
            commanded_rpy_err = [commanded_target_rpy[i] - tcp_rpy[i] for i in range(3)]
            desired_cmd_rpy_delta = [
                commanded_target_rpy[i] - desired_target_rpy[i] for i in range(3)
            ]

            if plt is not None and ax is not None and (now - last_plot_t) >= plot_period_s:
                update_plot(
                    ax,
                    desired_target_tcp,
                    desired_target_tcp_q,
                    commanded_target_tcp,
                    commanded_target_tcp_q,
                    desired_target_rpy,
                    commanded_target_rpy,
                )
                plt.tight_layout()
                plt.pause(0.001)
                last_plot_t = now

            print(
                " | ".join(
                    [
                        f"rx={rx_hz:6.1f}Hz",
                        f"src={addr[0]}:{addr[1]}",
                        f"state={teleop_state}",
                        f"mode={control_mode}",
                        f"active={int(teleop_active)}",
                        f"fresh={int(target_fresh)}",
                        f"ep_start={int(episode_start)}",
                        f"ep_end={int(episode_end)}",
                        f"pkt_age={packet_age_ms:6.1f}ms",
                        f"tgt_age={target_age_ms:6.1f}ms",
                        f"tcp=({float(tcp[0]):+.3f},{float(tcp[1]):+.3f},{float(tcp[2]):+.3f})",
                        f"des=({float(desired_target_tcp[0]):+.3f},{float(desired_target_tcp[1]):+.3f},{float(desired_target_tcp[2]):+.3f})",
                        f"cmd=({float(commanded_target_tcp[0]):+.3f},{float(commanded_target_tcp[1]):+.3f},{float(commanded_target_tcp[2]):+.3f})",
                        f"des_rpy=({desired_target_rpy[0]:+6.1f},{desired_target_rpy[1]:+6.1f},{desired_target_rpy[2]:+6.1f})",
                        f"cmd_rpy=({commanded_target_rpy[0]:+6.1f},{commanded_target_rpy[1]:+6.1f},{commanded_target_rpy[2]:+6.1f})",
                        f"cmd-des=({desired_cmd_pos_delta[0]:+6.3f},{desired_cmd_pos_delta[1]:+6.3f},{desired_cmd_pos_delta[2]:+6.3f})m",
                        f"cmd-des_rpy=({desired_cmd_rpy_delta[0]:+6.1f},{desired_cmd_rpy_delta[1]:+6.1f},{desired_cmd_rpy_delta[2]:+6.1f})",
                        f"des_err=({desired_rpy_err[0]:+6.1f},{desired_rpy_err[1]:+6.1f},{desired_rpy_err[2]:+6.1f})",
                        f"cmd_err=({commanded_rpy_err[0]:+6.1f},{commanded_rpy_err[1]:+6.1f},{commanded_rpy_err[2]:+6.1f})",
                        f"des_pos_err=({desired_pos_err[0]:+6.3f},{desired_pos_err[1]:+6.3f},{desired_pos_err[2]:+6.3f})m",
                        f"cmd_pos_err=({commanded_pos_err[0]:+6.3f},{commanded_pos_err[1]:+6.3f},{commanded_pos_err[2]:+6.3f})m",
                        f"action_space={action_space}",
                        f"|dpos|={dpos_n:.4f}m",
                        f"|drot|={drot_n:.4f}rad",
                        f"joint_tgt_max={joint_target_max:.4f}",
                        f"dq_max={dq_max:.4f}",
                        f"grip_cmd={grip_cmd:.3f}",
                        f"grip_state={gripper_state}",
                        f"grip_w={gripper_width:.3f}m",
                        f"manip={manip:.5f}",
                        f"faults={format_faults(faults)}",
                        f"stale={stale_ms:.1f}ms",
                    ]
                )
            )
            last_print_t = now

        except socket.timeout:
            now = time.monotonic()
            if plt is not None:
                plt.pause(0.001)
            if now - last_print_t >= print_period_s:
                if last_payload_t <= 0.0:
                    print("waiting_for_packets=true")
                else:
                    silent_ms = (now - last_payload_t) * 1e3
                    print(f"waiting_for_packets=true last_packet_ago_ms={silent_ms:.1f}")
                last_print_t = now
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
