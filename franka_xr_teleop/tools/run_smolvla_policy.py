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

from record_realsense_camera import import_dependencies as import_realsense_dependencies
from record_zed_camera import import_dependencies as import_zed_dependencies
from record_zed_camera import timestamp_to_ns as zed_timestamp_to_ns


OBS_STATE_KEY = "observation.state"
TOP_IMAGE_KEY = "observation.images.top"
THIRD_PERSON_IMAGE_KEY = "observation.images.third_person_d405"
ACTION_KEY = "action"
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7
SUPPORTED_PYTHON_MIN = (3, 12)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 14)
DEFAULT_ACTION_REFERENCE = "anchor_delta"
DEFAULT_POLICY_ANCHOR_RESET = "chunk"


def _ensure_supported_python() -> None:
    if "-h" in sys.argv or "--help" in sys.argv or "--list-cameras" in sys.argv:
        return
    version = sys.version_info[:3]
    if SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE:
        return
    current = ".".join(str(v) for v in version)
    min_supported = ".".join(str(v) for v in SUPPORTED_PYTHON_MIN)
    max_supported = ".".join(str(v) for v in (3, 13))
    raise RuntimeError(
        "run_smolvla_policy.py must be run with Python "
        f"{min_supported}-{max_supported}. Current interpreter: {current} "
        f"({sys.executable}). "
        "This local lerobot checkout uses draccus config parsing that is not "
        "working correctly under Python 3.14 here. Recreate or activate a "
        "Python 3.12/3.13 environment, then rerun the script."
    )


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

class ZedLeftCamera:
    def __init__(self, serial: int, resolution: str, fps: int) -> None:
        cv2, _np, sl = import_zed_dependencies()
        self._cv2 = cv2
        self._sl = sl
        self._zed = sl.Camera()
        self._image = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NONE
        if serial:
            init.set_from_serial_number(serial)

        err = self._zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        info = self._zed.get_camera_information()
        self.source = f"zed-left(serial={getattr(info, 'serial_number', serial or 'first')})"
        self._reported_width = None
        self._reported_height = None
        self._reported_fps = float(fps)
        if hasattr(info, "camera_configuration"):
            config = info.camera_configuration
            self._reported_width = getattr(config.resolution, "width", None) if hasattr(config, "resolution") else None
            self._reported_height = getattr(config.resolution, "height", None) if hasattr(config, "resolution") else None
            self._reported_fps = float(getattr(config, "fps", fps))
        self._serial = getattr(info, "serial_number", serial or None)
        self._model = str(getattr(info, "camera_model", ""))
        self._last_timestamp_ns: int | None = None

    def read_rgb(self, target_hw: tuple[int, int] | None = None) -> np.ndarray:
        err = self._zed.grab(self._runtime)
        if err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {err}")
        self._zed.retrieve_image(self._image, self._sl.VIEW.LEFT)
        self._last_timestamp_ns = zed_timestamp_to_ns(self._zed.get_timestamp(self._sl.TIME_REFERENCE.IMAGE))
        bgra = self._image.get_data()
        rgb = self._cv2.cvtColor(bgra, self._cv2.COLOR_BGRA2RGB)
        if target_hw is not None:
            h, w = target_hw
            if rgb.shape[:2] != (h, w):
                rgb = self._cv2.resize(rgb, (w, h), interpolation=self._cv2.INTER_AREA)
        return rgb

    def close(self) -> None:
        self._zed.close()

    def properties(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": "pyzed.sl",
            "reported_width": float(self._reported_width or 0),
            "reported_height": float(self._reported_height or 0),
            "reported_fps": self._reported_fps,
            "fourcc": "BGRA",
            "serial_number": self._serial,
            "camera_model": self._model,
            "zed_view": "LEFT",
            "zed_timestamp_ns": self._last_timestamp_ns,
        }


class RealSenseColorCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        cv2, np, rs = import_realsense_dependencies()
        self._cv2 = cv2
        self._np = np
        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if serial:
            self._config.enable_device(serial)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(self._config)
        self._width = width
        self._height = height
        self._fps = float(fps)

        dev = self._profile.get_device()
        self._serial = dev.get_info(rs.camera_info.serial_number)
        self._name = dev.get_info(rs.camera_info.name)
        self.source = f"realsense(serial={self._serial})"
        self._last_timestamp_ms: float | None = None

    def read_rgb(self, target_hw: tuple[int, int] | None = None) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame unavailable")
        self._last_timestamp_ms = float(color_frame.get_timestamp())
        bgr = self._np.asanyarray(color_frame.get_data())
        if bgr.shape[:2] != (self._height, self._width):
            bgr = self._cv2.resize(bgr, (self._width, self._height), interpolation=self._cv2.INTER_AREA)
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if target_hw is not None:
            h, w = target_hw
            if rgb.shape[:2] != (h, w):
                rgb = self._cv2.resize(rgb, (w, h), interpolation=self._cv2.INTER_AREA)
        return rgb

    def close(self) -> None:
        self._pipeline.stop()

    def properties(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": "pyrealsense2",
            "reported_width": float(self._width),
            "reported_height": float(self._height),
            "reported_fps": self._fps,
            "fourcc": "BGR8",
            "serial_number": self._serial,
            "camera_model": self._name,
            "realsense_timestamp_ms": self._last_timestamp_ms,
        }


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


def _validate_policy_features(policy: Any, image_keys: tuple[str, ...], print_full: bool = False) -> dict[str, Any]:
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

    for key in image_keys:
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

    supplied_live_keys = {OBS_STATE_KEY, *image_keys}
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
        "live_observation_keys": [OBS_STATE_KEY, *image_keys],
        "expected_state_dim": EXPECTED_STATE_DIM,
        "expected_action_dim": EXPECTED_ACTION_DIM,
        "image_features": sorted(image_features),
    }

    print("Policy feature compatibility:", flush=True)
    print(f"  live {OBS_STATE_KEY}: policy shape={summary['input_features'].get(OBS_STATE_KEY, {}).get('shape')}", flush=True)
    for key in image_keys:
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


def _load_action_stats(policy_path: Path) -> dict[str, Any] | None:
    stats_path = policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not stats_path.exists():
        return None
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None

    tensors = load_file(str(stats_path))

    def vec(name: str) -> list[float] | None:
        tensor = tensors.get(f"action.{name}")
        if tensor is None:
            return None
        return [float(v) for v in tensor.detach().cpu().reshape(-1).tolist()]

    q01 = vec("q01")
    q99 = vec("q99")
    if q01 is None or q99 is None or len(q01) < 7 or len(q99) < 7:
        return None
    return {
        "source": str(stats_path),
        "q01": q01[:7],
        "q99": q99[:7],
        "translation_q99_norm_m": float(np.linalg.norm(q99[:3])),
        "rotation_q99_norm_rad": float(np.linalg.norm(q99[3:6])),
    }


def _diagnose_action_with_info(
    action: np.ndarray,
    action_stats: dict[str, Any] | None,
    reject_translation_m: float,
    reject_rotation_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if action.shape[0] != 7:
        raise ValueError(f"Expected 7D action, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"Policy action contains non-finite values: {action.tolist()}")

    raw_translation_norm = float(np.linalg.norm(action[:3]))
    raw_rotation_norm = float(np.linalg.norm(action[3:6]))
    raw_gripper = float(action[6])
    gripper_clipped = not 0.0 <= raw_gripper <= 1.0

    if reject_translation_m > 0 and raw_translation_norm > reject_translation_m:
        raise ValueError(
            f"Policy translation norm {raw_translation_norm:.6f} m exceeds "
            f"--reject-translation-m={reject_translation_m:.6f}"
        )
    if reject_rotation_rad > 0 and raw_rotation_norm > reject_rotation_rad:
        raise ValueError(
            f"Policy rotation norm {raw_rotation_norm:.6f} rad exceeds "
            f"--reject-rotation-rad={reject_rotation_rad:.6f}"
        )

    action[6] = float(np.clip(action[6], 0.0, 1.0))
    outside_q01_q99: list[int] = []
    training_translation_q99_norm_m: float | None = None
    training_rotation_q99_norm_rad: float | None = None
    translation_above_training_q99_norm = False
    rotation_above_training_q99_norm = False
    if action_stats is not None:
        q01 = action_stats["q01"]
        q99 = action_stats["q99"]
        outside_q01_q99 = [
            idx for idx, value in enumerate(np.asarray(action, dtype=np.float64).reshape(-1))
            if idx < len(q01) and (float(value) < float(q01[idx]) or float(value) > float(q99[idx]))
        ]
        training_translation_q99_norm_m = float(action_stats["translation_q99_norm_m"])
        training_rotation_q99_norm_rad = float(action_stats["rotation_q99_norm_rad"])
        translation_above_training_q99_norm = raw_translation_norm > training_translation_q99_norm_m
        rotation_above_training_q99_norm = raw_rotation_norm > training_rotation_q99_norm_rad

    return action, {
        "raw_translation_norm_m": raw_translation_norm,
        "raw_rotation_norm_rad": raw_rotation_norm,
        "raw_gripper": raw_gripper,
        "sent_translation_norm_m": float(np.linalg.norm(action[:3])),
        "sent_rotation_norm_rad": float(np.linalg.norm(action[3:6])),
        "sent_gripper": float(action[6]),
        "outside_training_q01_q99_indices": outside_q01_q99,
        "training_translation_q99_norm_m": training_translation_q99_norm_m,
        "training_rotation_q99_norm_rad": training_rotation_q99_norm_rad,
        "translation_above_training_q99_norm": translation_above_training_q99_norm,
        "rotation_above_training_q99_norm": rotation_above_training_q99_norm,
        "translation_rejected": False,
        "rotation_rejected": False,
        "gripper_clipped": gripper_clipped,
    }


def _send_action(sock: socket.socket,
                 dst: tuple[str, int],
                 sequence_id: int,
                 action: np.ndarray,
                 enabled: bool,
                 action_reference: str,
                 episode_start: bool = False,
                 episode_end: bool = False) -> None:
    message = {
        "timestamp_ns": time.monotonic_ns(),
        "sequence_id": sequence_id,
        "enabled": enabled,
        "action_reference": action_reference,
        "episode_start": episode_start,
        "episode_end": episode_end,
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


def _policy_will_start_action_chunk(policy: Any) -> bool:
    checker = getattr(policy, "_check_get_actions_condition", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass

    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        action_queue = queues.get(ACTION_KEY)
        if action_queue is not None:
            return len(action_queue) == 0
    return False


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
    camera: Any,
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


def _list_realsense_devices() -> None:
    _cv2, _np, rs = import_realsense_dependencies()
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"realsense_device_count={len(devices)}")
    for index, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        firmware = dev.get_info(rs.camera_info.firmware_version)
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"realsense[{index}]: name={name} serial={serial} firmware={firmware} usb={usb_type}")


def _list_zed_devices() -> None:
    _cv2, _np, sl = import_zed_dependencies()
    devices = sl.Camera.get_device_list()
    print(f"zed_device_count={len(devices)}")
    for index, dev in enumerate(devices):
        serial = getattr(dev, "serial_number", None)
        model = getattr(dev, "camera_model", None)
        state = getattr(dev, "camera_state", None)
        print(f"zed[{index}]: serial={serial} model={model} state={state}")


def _list_available_cameras() -> int:
    _list_realsense_devices()
    _list_zed_devices()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available RealSense and ZED devices with serials, then exit.",
    )
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="")
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--obs-bind-ip", default="0.0.0.0")
    parser.add_argument("--obs-port", type=int, default=28081)
    parser.add_argument("--bridge-ip", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=28082)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--top-camera-backend",
        choices=["realsense", "zed-left"],
        default="realsense",
        help="SDK-backed source for observation.images.top.",
    )
    parser.add_argument(
        "--third-person-camera-backend",
        choices=["realsense", "zed-left"],
        default="zed-left",
        help=(
            "SDK-backed source for observation.images.third_person_d405. "
            "TODO: revisit these defaults and names once policy camera conventions are cleaned up."
        ),
    )
    parser.add_argument("--zed-serial", type=int, default=0, help="ZED serial number; 0 uses the first camera.")
    parser.add_argument(
        "--zed-resolution",
        default="HD720",
        choices=["VGA", "HD720", "HD1080", "HD2K"],
        help="ZED camera resolution requested from the SDK.",
    )
    parser.add_argument("--zed-fps", type=int, default=30, help="ZED camera FPS requested from the SDK.")
    parser.add_argument("--realsense-serial", default="", help="RealSense serial number; empty uses the first camera.")
    parser.add_argument("--realsense-color-width", type=int, default=1280)
    parser.add_argument("--realsense-color-height", type=int, default=720)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
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
        help="Optional JSONL path for raw/sent policy actions and action-distribution diagnostics.",
    )
    parser.add_argument(
        "--policy-action-reference",
        choices=["anchor_delta", "current_delta"],
        default=DEFAULT_ACTION_REFERENCE,
        help=(
            "How the bridge should interpret the policy action. "
            "anchor_delta matches teleop training data: action is an offset from "
            "the rollout anchor pose set by episode_start."
        ),
    )
    parser.add_argument(
        "--policy-anchor-reset",
        choices=["chunk", "run"],
        default=DEFAULT_POLICY_ANCHOR_RESET,
        help=(
            "Only used with --policy-action-reference=anchor_delta. "
            "chunk resets the bridge anchor whenever SmolVLA starts a new action chunk; "
            "run keeps one anchor for the whole policy process."
        ),
    )
    parser.add_argument(
        "--reject-translation-m",
        type=float,
        default=2.0,
        help="Reject policy actions above this translation norm. Set <=0 to disable.",
    )
    parser.add_argument(
        "--reject-rotation-rad",
        type=float,
        default=3.1416,
        help="Reject policy actions above this rotation-vector norm. Set <=0 to disable.",
    )
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="Send enabled zero actions without loading SmolVLA; useful for bridge smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_supported_python()
    args = parse_args()
    if args.list_cameras:
        return _list_available_cameras()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be > 0")
    if args.zed_fps <= 0:
        raise ValueError("--zed-fps must be > 0")
    if args.realsense_fps <= 0:
        raise ValueError("--realsense-fps must be > 0")
    if not args.zero_actions and args.policy_path is None:
        raise ValueError("--policy-path is required unless --zero-actions is set")
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
    action_stats: dict[str, Any] | None = None
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
            policy_path_obj = _resolve_policy_path(args.policy_path)
            policy_path = str(policy_path_obj)
            action_stats = _load_action_stats(policy_path_obj)
            policy = SmolVLAPolicy.from_pretrained(policy_path)
            policy.to(device)
            policy.eval()
            policy.reset()
            image_keys = (TOP_IMAGE_KEY, THIRD_PERSON_IMAGE_KEY)
            policy_features = _validate_policy_features(policy, image_keys, args.print_policy_features)
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            top_hw = _feature_image_shape(policy, TOP_IMAGE_KEY, (args.camera_height, args.camera_width))
            third_hw = _feature_image_shape(policy, THIRD_PERSON_IMAGE_KEY, (args.camera_height, args.camera_width))
            if args.top_camera_backend == "realsense":
                top_camera = RealSenseColorCamera(
                    args.realsense_serial,
                    args.realsense_color_width,
                    args.realsense_color_height,
                    args.realsense_fps,
                )
            else:
                top_camera = ZedLeftCamera(args.zed_serial, args.zed_resolution, args.zed_fps)
            if args.third_person_camera_backend == "realsense":
                third_person_camera = RealSenseColorCamera(
                    args.realsense_serial,
                    args.realsense_color_width,
                    args.realsense_color_height,
                    args.realsense_fps,
                )
            else:
                third_person_camera = ZedLeftCamera(args.zed_serial, args.zed_resolution, args.zed_fps)
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
            if action_stats is not None:
                print(
                    "Action diagnostics loaded: "
                    f"translation_q99_norm={action_stats['translation_q99_norm_m']:.6f}m, "
                    f"rotation_q99_norm={action_stats['rotation_q99_norm_rad']:.6f}rad",
                    flush=True,
                )
            else:
                print("Action diagnostics unavailable: policy action quantiles not found.", flush=True)
        else:
            torch = None
            prepare_observation_for_inference = None
            top_hw = (args.camera_height, args.camera_width)
            third_hw = (args.camera_height, args.camera_width)

        print(
            f"Streaming policy actions to udp://{args.bridge_ip}:{args.action_port} "
            f"from observations udp://{args.obs_bind_ip}:{args.obs_port} "
            f"at {args.rate_hz:.2f} Hz action_reference={args.policy_action_reference} "
            f"anchor_reset={args.policy_anchor_reset}",
            flush=True,
        )

        sequence_id = 0
        sent_episode_start = False
        while True:
            start = time.monotonic()
            obs = obs_rx.latest()
            if obs is None:
                time.sleep(min(period_s, 0.05))
                continue

            sequence_id += 1
            policy_chunk_start = False
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
                policy_chunk_start = _policy_will_start_action_chunk(policy)
                with torch.inference_mode():
                    action_tensor = policy.select_action(preprocess(frame))
                    action_tensor = postprocess(action_tensor)
                raw_action = action_tensor.squeeze(0).detach().cpu().numpy()

            action, action_info = _diagnose_action_with_info(
                raw_action,
                action_stats,
                args.reject_translation_m,
                args.reject_rotation_rad,
            )
            episode_start = not sent_episode_start or (
                args.policy_action_reference == "anchor_delta"
                and args.policy_anchor_reset == "chunk"
                and policy_chunk_start
            )
            _write_jsonl(action_log, {
                "timestamp_ns": time.monotonic_ns(),
                "sequence_id": sequence_id,
                "source": "zero_actions" if args.zero_actions else "policy",
                "task": args.task,
                "action_reference": args.policy_action_reference,
                "anchor_reset": args.policy_anchor_reset,
                "policy_chunk_start": policy_chunk_start,
                "episode_start": episode_start,
                "robot_observation_timestamp_ns": obs.get("timestamp_ns"),
                "raw_action": _jsonable_action(raw_action),
                "sent_action": _jsonable_action(action),
                "reject_translation_m": args.reject_translation_m,
                "reject_rotation_rad": args.reject_rotation_rad,
                **action_info,
            })
            _send_action(
                action_sock,
                dst,
                sequence_id,
                action,
                enabled=True,
                action_reference=args.policy_action_reference,
                episode_start=episode_start,
            )
            sent_episode_start = True

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
