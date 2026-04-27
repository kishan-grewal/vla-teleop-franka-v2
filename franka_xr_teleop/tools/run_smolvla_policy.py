#!/usr/bin/env python3
"""Run a SmolVLA policy and stream 7D Cartesian actions to the Franka bridge."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np


OBS_STATE_KEY = "observation.state"
TOP_IMAGE_KEY = "observation.images.top"
THIRD_PERSON_IMAGE_KEY = "observation.images.third_person_d405"
ACTION_KEY = "action"
LIVE_INPUT_KEYS = (OBS_STATE_KEY, TOP_IMAGE_KEY, THIRD_PERSON_IMAGE_KEY)
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7


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
        self.source = source
        self._cap = cv2.VideoCapture(_parse_camera_source(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source!r}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    def properties(self) -> dict[str, Any]:
        fourcc = int(self._cap.get(self._cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
        return {
            "source": self.source,
            "backend": self._cap.getBackendName() if hasattr(self._cap, "getBackendName") else None,
            "reported_width": float(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
            "reported_height": float(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_fps": float(self._cap.get(self._cv2.CAP_PROP_FPS)),
            "fourcc": fourcc_text,
        }


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


def _feature_shape(feature: Any) -> tuple[int, ...] | None:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _feature_type(feature: Any) -> str | None:
    feature_type = getattr(feature, "type", None)
    if feature_type is None and isinstance(feature, dict):
        feature_type = feature.get("type")
    if feature_type is None:
        return None
    return getattr(feature_type, "value", str(feature_type))


def _feature_summary(features: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "type": _feature_type(feature),
            "shape": list(_feature_shape(feature) or []),
        }
        for key, feature in sorted((features or {}).items())
    }


def _validate_policy_features(policy: Any, print_full: bool = False) -> dict[str, Any]:
    input_features = policy.config.input_features or {}
    output_features = policy.config.output_features or {}
    image_features = getattr(policy.config, "image_features", {}) or {}
    errors: list[str] = []

    state_shape = _feature_shape(input_features.get(OBS_STATE_KEY))
    if OBS_STATE_KEY not in input_features:
        errors.append(f"missing required input feature {OBS_STATE_KEY!r}")
    elif state_shape != (EXPECTED_STATE_DIM,):
        errors.append(
            f"{OBS_STATE_KEY!r} must have shape [{EXPECTED_STATE_DIM}], got {list(state_shape or [])}"
        )

    for key in (TOP_IMAGE_KEY, THIRD_PERSON_IMAGE_KEY):
        image_shape = _feature_shape(input_features.get(key))
        if key not in input_features:
            errors.append(f"missing required image feature {key!r}")
            continue
        if image_shape is None or len(image_shape) != 3:
            errors.append(f"{key!r} must have image shape [C,H,W], got {list(image_shape or [])}")
        elif image_shape[0] != 3:
            errors.append(f"{key!r} must have 3 color channels, got shape {list(image_shape)}")

    action_shape = _feature_shape(output_features.get(ACTION_KEY))
    if ACTION_KEY not in output_features:
        errors.append(f"missing required output feature {ACTION_KEY!r}")
    elif action_shape != (EXPECTED_ACTION_DIM,):
        errors.append(f"{ACTION_KEY!r} must have shape [{EXPECTED_ACTION_DIM}], got {list(action_shape or [])}")

    supplied_live_keys = set(LIVE_INPUT_KEYS)
    missing_live_image_keys = [
        key for key in image_features
        if key not in supplied_live_keys and not key.startswith("observation.images.empty_camera")
    ]
    if missing_live_image_keys:
        errors.append(
            "policy expects image feature(s) that this runner does not supply: "
            + ", ".join(repr(key) for key in missing_live_image_keys)
        )

    summary = {
        "input_features": _feature_summary(input_features),
        "output_features": _feature_summary(output_features),
        "live_observation_keys": list(LIVE_INPUT_KEYS),
        "expected_state_dim": EXPECTED_STATE_DIM,
        "expected_action_dim": EXPECTED_ACTION_DIM,
        "image_features": sorted(image_features),
    }

    print("Policy feature compatibility:", flush=True)
    for key in LIVE_INPUT_KEYS:
        print(f"  live {key}: policy shape={summary['input_features'].get(key, {}).get('shape')}", flush=True)
    print(f"  live {ACTION_KEY}: policy shape={summary['output_features'].get(ACTION_KEY, {}).get('shape')}", flush=True)
    if print_full:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if errors:
        raise ValueError("Policy feature compatibility check failed:\n- " + "\n- ".join(errors))
    return summary


def _feature_image_shape(policy: Any, key: str, fallback_hw: tuple[int, int]) -> tuple[int, int]:
    feature = policy.config.input_features.get(key)
    shape = _feature_shape(feature)
    if shape and len(shape) == 3:
        return int(shape[1]), int(shape[2])
    return fallback_hw


def _clamp_action_with_info(
    action: np.ndarray,
    max_translation_m: float,
    max_rotation_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if action.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"Policy action contains non-finite values: {action.tolist()}")

    raw_translation_norm = float(np.linalg.norm(action[:3]))
    raw_rotation_norm = float(np.linalg.norm(action[3:6]))
    raw_gripper = float(action[6])
    translation_clamped = False
    rotation_clamped = False
    gripper_clipped = not 0.0 <= raw_gripper <= 1.0

    t_norm = float(np.linalg.norm(action[:3]))
    if max_translation_m > 0 and t_norm > max_translation_m:
        action[:3] *= max_translation_m / max(t_norm, 1e-12)
        translation_clamped = True
    r_norm = float(np.linalg.norm(action[3:6]))
    if max_rotation_rad > 0 and r_norm > max_rotation_rad:
        action[3:6] *= max_rotation_rad / max(r_norm, 1e-12)
        rotation_clamped = True
    action[6] = float(np.clip(action[6], 0.0, 1.0))
    return action, {
        "raw_translation_norm_m": raw_translation_norm,
        "raw_rotation_norm_rad": raw_rotation_norm,
        "raw_gripper": raw_gripper,
        "clamped_translation_norm_m": float(np.linalg.norm(action[:3])),
        "clamped_rotation_norm_rad": float(np.linalg.norm(action[3:6])),
        "clamped_gripper": float(action[6]),
        "translation_clamped": translation_clamped,
        "rotation_clamped": rotation_clamped,
        "gripper_clipped": gripper_clipped,
    }


def _clamp_action(action: np.ndarray, max_translation_m: float, max_rotation_rad: float) -> np.ndarray:
    action, _info = _clamp_action_with_info(action, max_translation_m, max_rotation_rad)
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


def _write_jsonl(handle: TextIO | None, row: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    handle.flush()


def _jsonable_action(action: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(action, dtype=np.float64).reshape(-1)]


def _timestamped_preview_dir(root: Path) -> Path:
    return root.expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_preview_frame(path: Path, image_rgb: np.ndarray, label: str) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (12, 12), (min(bgr.shape[1] - 1, 620), 58), (0, 0, 0), thickness=-1)
    cv2.putText(
        bgr,
        label,
        (24, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to save preview frame to {path}")


def _capture_startup_preview(
    camera: OpenCVCamera,
    policy_key: str,
    target_hw: tuple[int, int],
    sample_count: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    sample_count = max(1, sample_count)
    start = time.monotonic()
    image = None
    for _ in range(sample_count):
        image = camera.read_rgb(target_hw)
    elapsed_s = max(time.monotonic() - start, 1e-9)
    assert image is not None

    preview_path: str | None = None
    if output_dir is not None:
        safe_key = policy_key.replace(".", "_")
        path = output_dir / f"{safe_key}.png"
        _save_preview_frame(path, image, f"{policy_key} source={camera.source}")
        preview_path = str(path)

    props = camera.properties()
    return {
        **props,
        "policy_key": policy_key,
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "observed_height": int(image.shape[0]),
        "observed_width": int(image.shape[1]),
        "observed_channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "read_samples": sample_count,
        "observed_read_fps": sample_count / elapsed_s,
        "preview_path": preview_path,
    }


def _print_camera_summary(summary: dict[str, Any]) -> None:
    print(
        "Camera identity: "
        f"{summary['policy_key']} <- source={summary['source']!r}, "
        f"reported={summary['reported_width']:.0f}x{summary['reported_height']:.0f}@"
        f"{summary['reported_fps']:.2f}fps, "
        f"observed={summary['observed_width']}x{summary['observed_height']}, "
        f"read_fps={summary['observed_read_fps']:.2f}, "
        f"preview={summary['preview_path']}",
        flush=True,
    )


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
    parser.add_argument(
        "--allow-default-camera-indices",
        action="store_true",
        help="Allow unsafe default OpenCV camera indices 0/1. Prefer explicit device paths or serial-backed sources.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("policy_previews"),
        help="Directory where startup camera identity previews are saved.",
    )
    parser.add_argument(
        "--skip-preview-frames",
        action="store_true",
        help="Do not save labelled startup preview frames.",
    )
    parser.add_argument(
        "--camera-preview-samples",
        type=int,
        default=5,
        help="Number of startup frames to read from each camera while estimating read FPS.",
    )
    parser.add_argument(
        "--print-policy-features",
        action="store_true",
        help="Print the full policy feature summary in addition to the compact compatibility check.",
    )
    parser.add_argument(
        "--log-actions-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for raw/clamped policy actions and clamp metadata.",
    )
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
    if not args.zero_actions and not args.allow_default_camera_indices:
        if args.top_camera == "0" and args.third_person_camera == "1":
            raise ValueError(
                "Default OpenCV camera indices are unsafe for policy deployment. "
                "Pass explicit --top-camera/--third-person-camera sources, or add "
                "--allow-default-camera-indices after verifying the startup preview frames."
            )
    if args.camera_preview_samples <= 0:
        raise ValueError("--camera-preview-samples must be > 0")

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
    action_log: TextIO | None = None

    try:
        if args.log_actions_jsonl is not None:
            args.log_actions_jsonl.expanduser().parent.mkdir(parents=True, exist_ok=True)
            action_log = args.log_actions_jsonl.expanduser().open("a", buffering=1)

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
            policy_features = _validate_policy_features(policy, args.print_policy_features)
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            top_hw = _feature_image_shape(policy, TOP_IMAGE_KEY, (args.camera_height, args.camera_width))
            third_hw = _feature_image_shape(
                policy,
                THIRD_PERSON_IMAGE_KEY,
                (args.camera_height, args.camera_width),
            )
            top_camera = OpenCVCamera(args.top_camera, args.camera_width, args.camera_height)
            third_person_camera = OpenCVCamera(
                args.third_person_camera,
                args.camera_width,
                args.camera_height,
            )
            preview_dir = None if args.skip_preview_frames else _timestamped_preview_dir(args.preview_dir)
            top_preview = _capture_startup_preview(
                top_camera,
                TOP_IMAGE_KEY,
                top_hw,
                args.camera_preview_samples,
                preview_dir,
            )
            third_preview = _capture_startup_preview(
                third_person_camera,
                THIRD_PERSON_IMAGE_KEY,
                third_hw,
                args.camera_preview_samples,
                preview_dir,
            )
            _print_camera_summary(top_preview)
            _print_camera_summary(third_preview)
            if preview_dir is not None:
                manifest_path = preview_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "policy_path": policy_path,
                            "task": args.task,
                            "policy_features": policy_features,
                            "cameras": [top_preview, third_preview],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"Saved camera preview manifest to {manifest_path}", flush=True)
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
                raw_action = np.zeros(7, dtype=np.float64)
            else:
                assert policy is not None
                assert preprocess is not None
                assert postprocess is not None
                assert top_camera is not None
                assert third_person_camera is not None
                assert torch is not None
                assert prepare_observation_for_inference is not None

                raw_observation = {
                    OBS_STATE_KEY: _robot_state_vector(obs),
                    TOP_IMAGE_KEY: top_camera.read_rgb(top_hw),
                    THIRD_PERSON_IMAGE_KEY: third_person_camera.read_rgb(third_hw),
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
                raw_action = action_tensor.squeeze(0).detach().cpu().numpy()

            action, clamp_info = _clamp_action_with_info(raw_action, args.max_translation_m, args.max_rotation_rad)
            _write_jsonl(action_log, {
                "timestamp_ns": time.monotonic_ns(),
                "sequence_id": sequence_id,
                "source": "zero_actions" if args.zero_actions else "policy",
                "task": args.task,
                "robot_observation_timestamp_ns": obs.get("timestamp_ns"),
                "raw_action": _jsonable_action(raw_action),
                "clamped_action": _jsonable_action(action),
                "max_translation_m": args.max_translation_m,
                "max_rotation_rad": args.max_rotation_rad,
                **clamp_info,
            })
            _send_action(action_sock, dst, sequence_id, action, enabled=True)

            elapsed = time.monotonic() - start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        return 0
    finally:
        obs_rx.stop()
        action_sock.close()
        if action_log is not None:
            action_log.close()
        if top_camera is not None:
            top_camera.close()
        if third_person_camera is not None:
            third_person_camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
