#!/usr/bin/env python3
"""Run a SmolVLA policy and stream 7D Cartesian actions to the Franka bridge."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class LatestRobotObservation:
    def __init__(self, bind_ip: str, port: int, timeout_s: float = 0.1) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(timeout_s)
        self._sock.bind((bind_ip, port))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            try:
                obs = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(obs, dict):
                with self._lock:
                    self._latest = obs


class OpenCVCamera:
    def __init__(self, source: str, width: int | None = None, height: int | None = None) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(_parse_camera_source(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source!r}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_rgb(self, target_hw: tuple[int, int] | None = None) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError("Camera frame read failed")
        if target_hw is not None:
            h, w = target_hw
            if bgr.shape[:2] != (h, w):
                bgr = self._cv2.resize(bgr, (w, h), interpolation=self._cv2.INTER_AREA)
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


def _parse_camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _resolve_lerobot_root(explicit_root: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root.expanduser())
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir.parents[2] / "lerobot")
    candidates.extend(parent / "lerobot" for parent in script_dir.parents)
    for candidate in candidates:
        if (candidate / "src" / "lerobot").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find lerobot/src. Pass --lerobot-root.")


def _ensure_lerobot_importable(lerobot_root: Path) -> None:
    lerobot_src = lerobot_root / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def _resolve_policy_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").exists():
        return path

    checkpoint_pretrained = path / "pretrained_model"
    if (checkpoint_pretrained / "config.json").exists():
        return checkpoint_pretrained

    last_pretrained = path / "checkpoints" / "last" / "pretrained_model"
    if (last_pretrained / "config.json").exists():
        return last_pretrained

    candidates = sorted((path / "checkpoints").glob("*/pretrained_model")) if (path / "checkpoints").exists() else []
    candidates = [candidate for candidate in candidates if (candidate / "config.json").exists()]
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"Could not find a loadable policy under {path}. "
        "Pass either a pretrained_model directory or a training output directory "
        "containing checkpoints/last/pretrained_model."
    )


def _robot_state_vector(obs: dict[str, Any]) -> np.ndarray:
    state = obs.get("robot_state", {})
    q = state.get("q", [])
    if len(q) != 7:
        raise ValueError("robot_state.q must contain 7 joints")
    gripper_width = float(state.get("gripper_width", 0.0))
    return np.asarray([*map(float, q), gripper_width], dtype=np.float32)


def _feature_image_shape(policy: Any, key: str, fallback_hw: tuple[int, int]) -> tuple[int, int]:
    feature = policy.config.input_features.get(key)
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape and len(shape) == 3:
        return int(shape[1]), int(shape[2])
    return fallback_hw


def _clamp_action(action: np.ndarray, max_translation_m: float, max_rotation_rad: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got shape {action.shape}")
    t_norm = float(np.linalg.norm(action[:3]))
    if max_translation_m > 0 and t_norm > max_translation_m:
        action[:3] *= max_translation_m / max(t_norm, 1e-12)
    r_norm = float(np.linalg.norm(action[3:6]))
    if max_rotation_rad > 0 and r_norm > max_rotation_rad:
        action[3:6] *= max_rotation_rad / max(r_norm, 1e-12)
    action[6] = float(np.clip(action[6], 0.0, 1.0))
    return action


def _send_action(sock: socket.socket,
                 dst: tuple[str, int],
                 sequence_id: int,
                 action: np.ndarray,
                 enabled: bool) -> None:
    message = {
        "timestamp_ns": time.monotonic_ns(),
        "sequence_id": sequence_id,
        "enabled": enabled,
        "action": [float(v) for v in action],
    }
    sock.sendto(json.dumps(message, separators=(",", ":")).encode("utf-8"), dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="")
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--obs-bind-ip", default="0.0.0.0")
    parser.add_argument("--obs-port", type=int, default=28081)
    parser.add_argument("--bridge-ip", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=28082)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--top-camera", default="0", help="OpenCV source for observation.images.top")
    parser.add_argument(
        "--third-person-camera",
        default="1",
        help="OpenCV source for observation.images.third_person_d405",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--max-translation-m", type=float, default=0.015)
    parser.add_argument("--max-rotation-rad", type=float, default=0.10)
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="Send enabled zero actions without loading SmolVLA; useful for bridge smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be > 0")
    if not args.zero_actions and args.policy_path is None:
        raise ValueError("--policy-path is required unless --zero-actions is set")

    obs_rx = LatestRobotObservation(args.obs_bind_ip, args.obs_port)
    obs_rx.start()
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.bridge_ip, args.action_port)
    period_s = 1.0 / args.rate_hz

    policy = None
    preprocess = None
    postprocess = None
    top_camera = None
    third_person_camera = None

    try:
        if not args.zero_actions:
            _ensure_lerobot_importable(_resolve_lerobot_root(args.lerobot_root))
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla import SmolVLAPolicy
            from lerobot.policies.utils import prepare_observation_for_inference

            device = torch.device(args.device)
            policy_path = str(_resolve_policy_path(args.policy_path))
            policy = SmolVLAPolicy.from_pretrained(policy_path)
            policy.to(device)
            policy.eval()
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            top_hw = _feature_image_shape(policy, "observation.images.top", (args.camera_height, args.camera_width))
            third_hw = _feature_image_shape(
                policy,
                "observation.images.third_person_d405",
                (args.camera_height, args.camera_width),
            )
            top_camera = OpenCVCamera(args.top_camera, args.camera_width, args.camera_height)
            third_person_camera = OpenCVCamera(
                args.third_person_camera,
                args.camera_width,
                args.camera_height,
            )
            print(f"Loaded SmolVLA policy from {policy_path}", flush=True)
        else:
            torch = None
            prepare_observation_for_inference = None
            top_hw = (args.camera_height, args.camera_width)
            third_hw = (args.camera_height, args.camera_width)

        print(
            f"Streaming policy actions to udp://{args.bridge_ip}:{args.action_port} "
            f"from observations udp://{args.obs_bind_ip}:{args.obs_port}",
            flush=True,
        )

        sequence_id = 0
        while True:
            start = time.monotonic()
            obs = obs_rx.latest()
            if obs is None:
                time.sleep(min(period_s, 0.05))
                continue

            sequence_id += 1
            if args.zero_actions:
                action = np.zeros(7, dtype=np.float64)
            else:
                assert policy is not None
                assert preprocess is not None
                assert postprocess is not None
                assert top_camera is not None
                assert third_person_camera is not None
                assert torch is not None
                assert prepare_observation_for_inference is not None

                raw_observation = {
                    "observation.state": _robot_state_vector(obs),
                    "observation.images.top": top_camera.read_rgb(top_hw),
                    "observation.images.third_person_d405": third_person_camera.read_rgb(third_hw),
                }
                frame = prepare_observation_for_inference(
                    raw_observation,
                    torch.device(args.device),
                    task=args.task,
                    robot_type=args.robot_type,
                )
                with torch.inference_mode():
                    action_tensor = policy.select_action(preprocess(frame))
                    action_tensor = postprocess(action_tensor)
                action = action_tensor.squeeze(0).detach().cpu().numpy()

            action = _clamp_action(action, args.max_translation_m, args.max_rotation_rad)
            _send_action(action_sock, dst, sequence_id, action, enabled=True)

            elapsed = time.monotonic() - start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        return 0
    finally:
        obs_rx.stop()
        action_sock.close()
        if top_camera is not None:
            top_camera.close()
        if third_person_camera is not None:
            third_person_camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
