#!/usr/bin/env python3
"""Run a LeRobot policy (SmolVLA, ACT, or Pi0) and stream absolute 7-joint targets to the Franka bridge."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
import select
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import numpy as np

from scipy.signal import butter, lfilter, lfilter_zi

from record_realsense_camera import import_dependencies as import_realsense_dependencies
from record_zed_camera import import_dependencies as import_zed_dependencies
from record_zed_camera import timestamp_to_ns as zed_timestamp_to_ns


OBS_STATE_KEY = "observation.state"
ACTION_KEY = "action"
JOINT_ACTION_DIM = 7
POLICY_ACTION_DIM = 8
POLICY_STATE_DIM = 8
SUPPORTED_PYTHON_MIN = (3, 12)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 14)
EXPOSURE_AUTO_SENTINEL = -1
EXPOSURE_MIN = 0
EXPOSURE_MAX = 100
REHOME_REQUEST_REPEAT_PACKETS = 10
JOINT_LIMIT_MARGIN_RAD = 0.02
PANDA_JOINT_LOWER_LIMITS_RAD = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_JOINT_UPPER_LIMITS_RAD = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)


def _load_smolvla_policy_class() -> Any:
    from lerobot.policies.smolvla import SmolVLAPolicy
    return SmolVLAPolicy


def _load_act_policy_class() -> Any:
    from lerobot.policies.act import ACTPolicy
    return ACTPolicy


def _load_pi0_policy_class() -> Any:
    from lerobot.policies.pi0 import PI0Policy
    return PI0Policy


# Registry of supported policy types. The value is a thin loader so the heavy
# imports happen only when the user actually picks that policy. Add new
# policies here to expose them through --policy-type.
POLICY_REGISTRY: dict[str, Callable[[], Any]] = {
    "smolvla": _load_smolvla_policy_class,
    "act": _load_act_policy_class,
    "pi0": _load_pi0_policy_class,
}

# RTC-compatible policy types (flow-matching based).
RTC_COMPATIBLE_POLICY_TYPES: set[str] = {"smolvla", "pi0"}


with contextlib.suppress(ImportError):
    import termios
    import tty

if "termios" not in globals():
    termios = None  # type: ignore[assignment]
if "tty" not in globals():
    tty = None  # type: ignore[assignment]


@dataclass
class _CameraSource:
    obs_key: str
    camera: Any  # RealSenseColorCamera or ZedStereoCamera
    zed_view: str | None  # None for RealSense; "left"/"right" for ZED
    target_hw: tuple[int, int]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _obs_keys_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for cam_cfg in config.get("cameras", []):
        if not cam_cfg.get("enabled", True):
            continue
        backend = str(cam_cfg.get("backend", "realsense"))
        if backend == "realsense":
            if obs_key := cam_cfg.get("obs_key"):
                keys.append(obs_key)
        elif backend == "zed":
            for attr in ("obs_key_left", "obs_key_right"):
                if obs_key := cam_cfg.get(attr):
                    keys.append(obs_key)
    return tuple(keys)


def _build_camera_sources(
    config: dict[str, Any],
    policy: Any | None,
    fallback_hw: tuple[int, int],
) -> list[_CameraSource]:
    sources: list[_CameraSource] = []
    zed_objects: dict[str, Any] = {}
    for cam_cfg in config.get("cameras", []):
        if not cam_cfg.get("enabled", True):
            continue
        backend = str(cam_cfg.get("backend", "realsense"))
        cam_id = str(cam_cfg.get("id", ""))
        if backend == "realsense":
            obs_key = cam_cfg.get("obs_key")
            if not obs_key:
                continue
            hw = _feature_image_shape(policy, obs_key, fallback_hw) if policy is not None else fallback_hw
            camera = RealSenseColorCamera(
                str(cam_cfg.get("serial", "")),
                int(cam_cfg.get("color_width", 1280)),
                int(cam_cfg.get("color_height", 720)),
                int(cam_cfg.get("fps", 30)),
            )
            sources.append(_CameraSource(obs_key=obs_key, camera=camera, zed_view=None, target_hw=hw))
        elif backend == "zed":
            if cam_id not in zed_objects:
                import types as _types
                zed_cam = ZedStereoCamera(
                    int(cam_cfg.get("serial", 0)),
                    str(cam_cfg.get("resolution", "HD720")),
                    int(cam_cfg.get("fps", 30)),
                )
                zed_cam.configure_exposure(_types.SimpleNamespace(
                    exposure=cam_cfg.get("exposure", 60),
                    auto_exposure=bool(cam_cfg.get("auto_exposure", False)),
                ))
                zed_objects[cam_id] = zed_cam
            zed_cam = zed_objects[cam_id]
            for attr, view in (("obs_key_left", "left"), ("obs_key_right", "right")):
                obs_key = cam_cfg.get(attr)
                if obs_key:
                    hw = _feature_image_shape(policy, obs_key, fallback_hw) if policy is not None else fallback_hw
                    sources.append(_CameraSource(obs_key=obs_key, camera=zed_cam, zed_view=view, target_hw=hw))
    return sources


def _read_images(sources: list[_CameraSource]) -> dict[str, np.ndarray]:
    """Read one frame from each camera source, batching ZED grabs per camera object."""
    images: dict[str, np.ndarray] = {}
    zed_groups: dict[int, list[_CameraSource]] = {}
    for source in sources:
        if source.zed_view is not None:
            zed_groups.setdefault(id(source.camera), []).append(source)
        else:
            images[source.obs_key] = source.camera.read_rgb(source.target_hw)
    for zed_sources in zed_groups.values():
        results = zed_sources[0].camera.read_views_rgb(
            *((s.zed_view, s.target_hw) for s in zed_sources)
        )
        for source, image in zip(zed_sources, results):
            images[source.obs_key] = image
    return images


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
        "run_lerobot_policy.py must be run with Python "
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


class KeyboardMonitor:
    def __init__(self) -> None:
        self._enabled = bool(sys.stdin.isatty() and termios is not None and tty is not None)
        self._fd = sys.stdin.fileno() if self._enabled else None
        self._saved_attrs: list[Any] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled or self._fd is None:
            return
        self._saved_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def stop(self) -> None:
        if self._saved_attrs is None or self._fd is None:
            return
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        self._saved_attrs = None

    def poll(self) -> list[str]:
        if not self._enabled or self._fd is None:
            return []
        chars: list[str] = []
        while True:
            readable, _writeable, _exceptional = select.select([self._fd], [], [], 0.0)
            if not readable:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            chars.append(ch)
        return chars


class LiveTuningReceiver:
    def __init__(self, port: int, defaults: dict[str, float]) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.05)
        self._sock.bind(("127.0.0.1", port))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._values: dict[str, float] = dict(defaults)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close()

    def get(self, key: str) -> float:
        with self._lock:
            return self._values[key]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            with self._lock:
                for k, v in data.items():
                    if k in self._values:
                        self._values[k] = float(v)
                        print(f"Tuning: {k}={v}", flush=True)


class ZedStereoCamera:
    def __init__(self, serial: int, resolution: str, fps: int) -> None:
        cv2, _np, sl = import_zed_dependencies()
        self._cv2 = cv2
        self._sl = sl
        self._zed = sl.Camera()
        self._left_image = sl.Mat()
        self._right_image = sl.Mat()
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
        self.source = f"zed-stereo(serial={getattr(info, 'serial_number', serial or 'first')})"
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

    def _resize_rgb(self, rgb: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
        if target_hw is None:
            return rgb
        h, w = target_hw
        if rgb.shape[:2] == (h, w):
            return rgb
        return self._cv2.resize(rgb, (w, h), interpolation=self._cv2.INTER_AREA)

    def read_views_rgb(self, *requests: tuple[str, tuple[int, int] | None]) -> list[np.ndarray]:
        err = self._zed.grab(self._runtime)
        if err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {err}")
        self._zed.retrieve_image(self._left_image, self._sl.VIEW.LEFT)
        self._zed.retrieve_image(self._right_image, self._sl.VIEW.RIGHT)
        self._last_timestamp_ns = zed_timestamp_to_ns(self._zed.get_timestamp(self._sl.TIME_REFERENCE.IMAGE))
        left_rgb = self._cv2.cvtColor(self._left_image.get_data(), self._cv2.COLOR_BGRA2RGB)
        right_rgb = self._cv2.cvtColor(self._right_image.get_data(), self._cv2.COLOR_BGRA2RGB)

        outputs: list[np.ndarray] = []
        for view, target_hw in requests:
            if view == "left":
                outputs.append(self._resize_rgb(left_rgb, target_hw))
            elif view == "right":
                outputs.append(self._resize_rgb(right_rgb, target_hw))
            else:
                raise ValueError(f"Unsupported ZED view {view!r}; expected 'left' or 'right'")
        return outputs

    def configure_exposure(self, args: argparse.Namespace) -> Optional[int]:
        want_auto_exposure = args.auto_exposure or args.exposure == EXPOSURE_AUTO_SENTINEL
        if want_auto_exposure:
            err = self._zed.set_camera_settings(self._sl.VIDEO_SETTINGS.AEC_AGC, 1)
            if err != self._sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to enable ZED auto exposure: {err}")
            err = self._zed.set_camera_settings(self._sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_AUTO_SENTINEL)
            if err != self._sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to reset ZED exposure to auto: {err}")
            aec_agc = self.get_camera_setting(self._sl.VIDEO_SETTINGS.AEC_AGC, "AEC_AGC state")
            if aec_agc != 1:
                raise RuntimeError(f"Requested ZED auto exposure, but camera reported AEC_AGC={aec_agc}")
            exposure = self.get_camera_setting(self._sl.VIDEO_SETTINGS.EXPOSURE, "exposure after enabling auto mode")
            return exposure

        if args.exposure is None:
            return None

        if not EXPOSURE_MIN <= args.exposure <= EXPOSURE_MAX:
            raise ValueError(
                "--exposure must be -1 for auto exposure, or within the ZED SDK "
                f"documented manual range [{EXPOSURE_MIN}, {EXPOSURE_MAX}]; got {args.exposure}"
            )

        err = self._zed.set_camera_settings(self._sl.VIDEO_SETTINGS.AEC_AGC, 0)
        if err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to disable ZED auto exposure before manual exposure set: {err}")
        err = self._zed.set_camera_settings(self._sl.VIDEO_SETTINGS.EXPOSURE, args.exposure)
        if err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to set ZED exposure to {args.exposure}: {err}")

        aec_agc = self.get_camera_setting(self._sl.VIDEO_SETTINGS.AEC_AGC, "AEC_AGC state")
        if aec_agc != 0:
            raise RuntimeError(
                f"Requested manual ZED exposure {args.exposure}, but camera reported AEC_AGC={aec_agc}"
            )
        exposure = self.get_camera_setting(self._sl.VIDEO_SETTINGS.EXPOSURE, "exposure after setting it")
        if exposure != args.exposure:
            raise RuntimeError(
                f"Requested ZED exposure {args.exposure}, but camera reported exposure {exposure}"
            )
        return exposure

    def get_camera_setting(self, setting: Any, label: str) -> int:
        read_err, value = self._zed.get_camera_settings(setting)
        if read_err != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to read ZED {label}: {read_err}")
        return value

    def close(self) -> None:
        self._zed.close()

    def properties(self, view: str) -> dict[str, Any]:
        if view not in {"left", "right"}:
            raise ValueError(f"Unsupported ZED view {view!r}; expected 'left' or 'right'")
        return {
            "source": f"{self.source}:{view}",
            "backend": "pyzed.sl",
            "reported_width": float(self._reported_width or 0),
            "reported_height": float(self._reported_height or 0),
            "reported_fps": self._reported_fps,
            "fourcc": "BGRA",
            "serial_number": self._serial,
            "camera_model": self._model,
            "zed_view": view.upper(),
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


def _robot_state_vector(obs: dict[str, Any], expected_dim: int) -> np.ndarray:
    if expected_dim != POLICY_STATE_DIM:
        raise ValueError(
            f"Unsupported policy state dimension {expected_dim}; "
            f"runner expects exactly [{POLICY_STATE_DIM}] = 7 joint angles + 1 gripper width"
        )
    state = obs.get("robot_state", {})
    q = state.get("q", [])
    if len(q) != 7:
        raise ValueError("robot_state.q must contain 7 joints")
    gripper_width = float(state.get("gripper_width", 0.0))
    values = [*map(float, q), gripper_width]
    return np.asarray(values, dtype=np.float32)


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
    elif state_shape is None or len(state_shape) != 1:
        errors.append(f"{OBS_STATE_KEY!r} must have shape [D], got {list(state_shape or [])}")
    elif state_shape != (POLICY_STATE_DIM,):
        errors.append(
            f"{OBS_STATE_KEY!r} must have shape [{POLICY_STATE_DIM}] "
            "(7 joint angles + 1 gripper width), "
            f"got {list(state_shape)}"
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
    elif action_shape != (POLICY_ACTION_DIM,):
        errors.append(f"{ACTION_KEY!r} must have shape [{POLICY_ACTION_DIM}], got {list(action_shape or [])}")

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
        "expected_state_dim": POLICY_STATE_DIM,
        "expected_action_dim": POLICY_ACTION_DIM,
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


def _action_sequence_from_chunk(action_chunk: Any, name: str) -> Any:
    """Return a single action sequence shaped (time_steps, action_dim)."""
    ndim = getattr(action_chunk, "ndim", None)
    if ndim == 3:
        if int(action_chunk.shape[0]) != 1:
            raise ValueError(f"{name} has batch size {action_chunk.shape[0]}; only batch size 1 is supported")
        return action_chunk.squeeze(0)
    if ndim == 2:
        return action_chunk
    raise ValueError(f"{name} must have shape (1, T, A) or (T, A), got {tuple(action_chunk.shape)}")


def _clamp_action_with_info(action: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if action.shape[0] != POLICY_ACTION_DIM:
        raise ValueError(f"Expected {POLICY_ACTION_DIM}D joint+gripper action, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"Policy action contains non-finite values: {action.tolist()}")

    joint_positions = action[:JOINT_ACTION_DIM].copy()
    raw_gripper = float(action[JOINT_ACTION_DIM])
    raw_joint_min = float(np.min(joint_positions))
    raw_joint_max = float(np.max(joint_positions))
    lower = PANDA_JOINT_LOWER_LIMITS_RAD + JOINT_LIMIT_MARGIN_RAD
    upper = PANDA_JOINT_UPPER_LIMITS_RAD - JOINT_LIMIT_MARGIN_RAD
    clipped_mask = np.logical_or(joint_positions < lower, joint_positions > upper)
    joint_positions = np.clip(joint_positions, lower, upper)
    clamped_gripper = 1.0 if raw_gripper >= 0.5 else 0.0
    return joint_positions, clamped_gripper, {
        "joint_limit_margin_rad": JOINT_LIMIT_MARGIN_RAD,
        "raw_joint_min_rad": raw_joint_min,
        "raw_joint_max_rad": raw_joint_max,
        "clamped_joint_min_rad": float(np.min(joint_positions)),
        "clamped_joint_max_rad": float(np.max(joint_positions)),
        "joint_limit_clipped": bool(np.any(clipped_mask)),
        "joint_limit_clipped_indices": [int(i) for i, clipped in enumerate(clipped_mask) if clipped],
        "raw_gripper": raw_gripper,
        "clamped_gripper": clamped_gripper,
        "gripper_binarized": raw_gripper != clamped_gripper,
    }


def _send_action(sock: socket.socket,
                 dst: tuple[str, int],
                 sequence_id: int,
                 joint_positions_rad: np.ndarray,
                 gripper_command: float,
                 enabled: bool,
                 operator_request_id: int = 0,
                 request_rehome: bool = False) -> None:
    message = {
        "timestamp_ns": time.monotonic_ns(),
        "sequence_id": sequence_id,
        "enabled": enabled,
        "action_space": "joint_position_absolute",
        "joint_positions_rad": [float(v) for v in joint_positions_rad],
        "gripper_command": float(np.clip(gripper_command, 0.0, 1.0)),
    }
    if operator_request_id > 0:
        message["operator_request_id"] = int(operator_request_id)
    if request_rehome:
        message["request_rehome"] = True
    sock.sendto(json.dumps(message, separators=(",", ":")).encode("utf-8"), dst)


def _write_jsonl(handle: TextIO | None, row: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    handle.flush()


def _jsonable_action(action: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(action, dtype=np.float64).reshape(-1)]


def _model_output_log_fields(raw_action: np.ndarray) -> dict[str, Any]:
    action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if action.shape[0] != POLICY_ACTION_DIM:
        return {
            "model_output_unexpected_action_dim": int(action.shape[0]),
            "model_output_expected_action_dim": POLICY_ACTION_DIM,
        }
    raw_joints = action[:JOINT_ACTION_DIM]
    raw_gripper = float(action[JOINT_ACTION_DIM])
    return {
        "model_raw_joint_positions_rad": [float(v) for v in raw_joints],
        "model_raw_gripper": raw_gripper,
    }


def _current_joint_positions(obs: dict[str, Any]) -> np.ndarray:
    q = obs.get("robot_state", {}).get("q", [])
    if not isinstance(q, list) or len(q) != JOINT_ACTION_DIM:
        raise ValueError("robot_state.q must contain 7 joint positions for joint-space policy control")
    joint_positions = np.asarray(q, dtype=np.float64)
    if not np.isfinite(joint_positions).all():
        raise ValueError(f"robot_state.q contains non-finite values: {q}")
    return joint_positions


def _current_hold_action(
    obs: dict[str, Any],
    gripper_command_override: Optional[float] = None,
) -> np.ndarray:
    joint_positions = _current_joint_positions(obs)
    if gripper_command_override is None:
        gripper_state = str(obs.get("robot_state", {}).get("gripper_state", "OPEN")).upper()
        gripper_command = 1.0 if gripper_state in {"CLOSE", "HOLD"} else 0.0
    else:
        gripper_command = float(np.clip(gripper_command_override, 0.0, 1.0))
    return np.concatenate([joint_positions, np.asarray([gripper_command], dtype=np.float64)])


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


def _preview_summary_from_image(
    properties: dict[str, Any],
    policy_key: str,
    target_hw: tuple[int, int],
    sample_count: int,
    elapsed_s: float,
    image: np.ndarray,
    output_dir: Path | None,
) -> dict[str, Any]:
    preview_path: str | None = None
    if output_dir is not None:
        safe_key = policy_key.replace(".", "_")
        path = output_dir / f"{safe_key}.png"
        _save_preview_frame(path, image, f"{policy_key} source={properties['source']}")
        preview_path = str(path)

    return {
        **properties,
        "policy_key": policy_key,
        "target_height": int(target_hw[0]),
        "target_width": int(target_hw[1]),
        "observed_height": int(image.shape[0]),
        "observed_width": int(image.shape[1]),
        "observed_channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "read_samples": sample_count,
        "observed_read_fps": sample_count / max(elapsed_s, 1e-9),
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


def _measure_inference_delay(
    policy: Any,
    preprocess: Any,
    postprocess: Any,
    prepare_observation_for_inference: Any,
    camera_sources: list[_CameraSource],
    obs: dict[str, Any],
    policy_state_dim: int,
    device: Any,
    task: str,
    robot_type: str,
    rate_hz: float,
    warmup_iters: int = 3,
    measure_iters: int = 5,
) -> int:
    """Measure startup policy latency and convert to action steps at the given rate."""
    import torch

    raw_observation = {
        OBS_STATE_KEY: _robot_state_vector(obs, policy_state_dim),
        **_read_images(camera_sources),
    }
    frame = prepare_observation_for_inference(
        raw_observation,
        device,
        task=task,
        robot_type=robot_type,
    )

    # Warmup
    for _ in range(warmup_iters):
        with torch.inference_mode():
            _ = policy.predict_action_chunk(preprocess(frame))

    # Measure
    times: list[float] = []
    for _ in range(measure_iters):
        start = time.monotonic()
        with torch.inference_mode():
            _ = policy.predict_action_chunk(preprocess(frame))
        times.append(time.monotonic() - start)

    median_s = float(np.median(times))
    step_period_s = 1.0 / rate_hz
    inference_delay = max(1, int(np.ceil(median_s / step_period_s)))
    print(
        f"RTC inference delay measurement: median={median_s * 1000:.1f}ms "
        f"step_period={step_period_s * 1000:.1f}ms inference_delay={inference_delay} steps",
        flush=True,
    )
    return inference_delay


def _sequence_length(value: Any | None) -> int | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) > 0 else 0
    with contextlib.suppress(TypeError):
        return len(value)
    return None


class RTCActionProducer:
    """Background chunk producer for RTC deployment.

    The main loop stays responsible for sending one action packet per tick.
    This producer observes, infers, postprocesses, and merges chunks into the
    shared ActionQueue without blocking UDP action streaming.
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocess: Any,
        postprocess: Any,
        prepare_observation_for_inference: Any,
        camera_sources: list[_CameraSource],
        obs_rx: LatestRobotObservation,
        action_queue: Any,
        policy_state_dim: int,
        device: Any,
        task: str,
        robot_type: str,
        period_s: float,
        initial_inference_delay: int,
        execution_horizon: int,
        refill_threshold: int | None,
    ) -> None:
        self._policy = policy
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._prepare_observation_for_inference = prepare_observation_for_inference
        self._camera_sources = camera_sources
        self._obs_rx = obs_rx
        self._action_queue = action_queue
        self._policy_state_dim = policy_state_dim
        self._device = device
        self._task = task
        self._robot_type = robot_type
        self._period_s = period_s
        self._execution_horizon = max(1, int(execution_horizon))
        self._explicit_refill_threshold = refill_threshold
        self._current_delay_steps = max(1, int(initial_inference_delay))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active = True
        self._busy = False
        self._generation = 0
        self._requested_refill_reason: str | None = "initial"
        self._thread = threading.Thread(target=self._run, name="RTCActionProducer", daemon=True)

        self._inference_count = 0
        self._skipped_merge_count = 0
        self._underrun_count = 0
        self._last_refill_reason: str | None = None
        self._last_total_ms: float | None = None
        self._last_model_ms: float | None = None
        self._last_merge_delay_steps: int | None = None
        self._last_queue_advance_steps: int | None = None
        self._last_prev_leftover_len: int | None = None
        self._last_action_index_before_inference: int | None = None
        self._last_error_traceback: str | None = None

    def start(self) -> None:
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join()

    def set_active(self, active: bool, *, clear_queue: bool, reason: str) -> None:
        with self._lock:
            self._active = active
            self._generation += 1
            if clear_queue:
                self._action_queue.clear()
            self._requested_refill_reason = reason if active else None
        self._wake.set()

    def request_inference(self, reason: str) -> None:
        with self._lock:
            if self._active:
                self._requested_refill_reason = reason
        self._wake.set()

    def notify_queue_underrun(self) -> None:
        with self._lock:
            self._underrun_count += 1
            if self._active:
                self._requested_refill_reason = "queue_empty_after_pop"
        self._wake.set()

    def refill_threshold(self) -> int:
        with self._lock:
            return self._refill_threshold_locked()

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._lock:
                if not self._busy:
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def raise_if_failed(self) -> None:
        with self._lock:
            error_traceback = self._last_error_traceback
        if error_traceback is not None:
            raise RuntimeError(f"RTC action producer failed:\n{error_traceback}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            queue_size = self._action_queue.qsize()
            original_leftover_len = _sequence_length(self._action_queue.get_left_over())
            processed_leftover_len = _sequence_length(self._action_queue.get_processed_left_over())
            return {
                "rtc_producer_active": self._active,
                "rtc_producer_busy": self._busy,
                "rtc_queue_size": queue_size,
                "rtc_original_leftover_len": original_leftover_len,
                "rtc_processed_leftover_len": processed_leftover_len,
                "rtc_refill_threshold": self._refill_threshold_locked(),
                "rtc_execution_horizon": self._execution_horizon,
                "rtc_current_inference_delay": self._current_delay_steps,
                "rtc_last_merge_delay": self._last_merge_delay_steps,
                "rtc_last_queue_advance_steps": self._last_queue_advance_steps,
                "rtc_last_chunk_total_ms": self._last_total_ms,
                "rtc_last_model_inference_ms": self._last_model_ms,
                "rtc_last_refill_reason": self._last_refill_reason,
                "rtc_last_prev_leftover_len": self._last_prev_leftover_len,
                "rtc_last_action_index_before_inference": self._last_action_index_before_inference,
                "rtc_inference_count": self._inference_count,
                "rtc_skipped_merge_count": self._skipped_merge_count,
                "rtc_queue_underrun_count": self._underrun_count,
                "rtc_producer_error": self._last_error_traceback is not None,
            }

    def _refill_threshold_locked(self) -> int:
        if self._explicit_refill_threshold is not None:
            return self._explicit_refill_threshold
        return self._current_delay_steps + self._execution_horizon

    def _run(self) -> None:
        poll_s = min(max(self._period_s * 0.5, 0.005), 0.05)
        while not self._stop.is_set():
            self._wake.wait(timeout=poll_s)
            self._wake.clear()
            if self._stop.is_set():
                break

            with self._lock:
                if not self._active:
                    continue

                queue_size = self._action_queue.qsize()
                refill_threshold = self._refill_threshold_locked()
                forced_reason = self._requested_refill_reason
                if forced_reason is not None:
                    refill_reason = forced_reason
                    self._requested_refill_reason = None
                elif queue_size <= refill_threshold:
                    refill_reason = "queue_low"
                else:
                    continue

                obs = self._obs_rx.latest()
                if obs is None:
                    self._requested_refill_reason = refill_reason
                    continue

                generation = self._generation
                planned_delay_steps = self._current_delay_steps
                action_index_before_inference = self._action_queue.get_action_index()
                prev_actions = self._action_queue.get_left_over()
                prev_leftover_len = _sequence_length(prev_actions)
                self._last_refill_reason = refill_reason
                self._last_prev_leftover_len = prev_leftover_len
                self._last_action_index_before_inference = action_index_before_inference
                self._busy = True

            try:
                start = time.monotonic()
                raw_observation = {
                    OBS_STATE_KEY: _robot_state_vector(obs, self._policy_state_dim),
                    **_read_images(self._camera_sources),
                }
                frame = self._prepare_observation_for_inference(
                    raw_observation,
                    self._device,
                    task=self._task,
                    robot_type=self._robot_type,
                )
                preprocessed_frame = self._preprocess(frame)
                model_start = time.monotonic()
                action_chunk = self._policy.predict_action_chunk(
                    preprocessed_frame,
                    inference_delay=planned_delay_steps,
                    prev_chunk_left_over=prev_actions,
                )
                model_elapsed_s = time.monotonic() - model_start
                processed_action_chunk = self._postprocess(action_chunk.clone())
                original_actions = _action_sequence_from_chunk(action_chunk, "RTC action chunk").clone()
                processed_actions = _action_sequence_from_chunk(
                    processed_action_chunk,
                    "postprocessed RTC action chunk",
                )
                total_elapsed_s = time.monotonic() - start
                merge_delay_steps = max(1, int(math.ceil(total_elapsed_s / self._period_s)))

                with self._lock:
                    queue_advance_steps = max(
                        0,
                        self._action_queue.get_action_index() - action_index_before_inference,
                    )
                    if self._active and generation == self._generation and not self._stop.is_set():
                        self._action_queue.merge(
                            original_actions,
                            processed_actions,
                            merge_delay_steps,
                        )
                        self._current_delay_steps = merge_delay_steps
                        self._last_total_ms = total_elapsed_s * 1000.0
                        self._last_model_ms = model_elapsed_s * 1000.0
                        self._last_merge_delay_steps = merge_delay_steps
                        self._last_queue_advance_steps = queue_advance_steps
                        self._inference_count += 1
                        self._requested_refill_reason = None
                    else:
                        self._skipped_merge_count += 1
            except Exception:
                with self._lock:
                    self._last_error_traceback = traceback.format_exc()
                self._stop.set()
            finally:
                with self._lock:
                    self._busy = False


def _validate_rtc_flags(args: argparse.Namespace) -> None:
    """Error if RTC-specific flags are set without --use-rtc."""
    if args.use_rtc:
        return
    rtc_flags: dict[str, Any] = {
        "--rtc-execution-horizon": (args.rtc_execution_horizon, 0),
        "--rtc-max-guidance-weight": (args.rtc_max_guidance_weight, 10.0),
        "--rtc-attention-schedule": (args.rtc_attention_schedule, "EXP"),
        "--rtc-inference-delay": (args.rtc_inference_delay, 0),
        "--rtc-refill-threshold": (args.rtc_refill_threshold, 0),
    }
    non_default = [flag for flag, (value, default) in rtc_flags.items() if value != default]
    if non_default:
        raise ValueError(
            f"RTC flags {', '.join(non_default)} were set but --use-rtc is not enabled. "
            "Either add --use-rtc or remove the RTC-specific flags."
        )

class GripperHoldRequirement:
    def __init__(self, command_count_threshold: int):
        self.command_count_threshold = command_count_threshold
        self.latched_command = 0

        self.command_index = 0
        self.latched_command_index = 0

    def update(self, command: float) -> float:
        self.command_index += 1
        command = 1.0 if command > 0.5 else 0.0
        if command == self.latched_command:
            self.latched_command_index = self.command_index
            return self.latched_command

        if self.command_index - self.latched_command_index > self.command_count_threshold:
            self.latched_command = command
            self.latched_command_index = self.command_index
        return self.latched_command
    
    def get(self) -> float:
        return self.latched_command

    def reset(self) -> None:
        self.command_index = 0
        self.latched_command_index = 0
        self.latched_command = 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available RealSense and ZED devices with serials, then exit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML config in data_collection.yaml format. Enabled cameras with obs_key "
            "(or obs_key_left / obs_key_right for ZED) drive camera setup and policy "
            "observation key mapping. Robot bind_ip / port override --obs-bind-ip / --obs-port."
        ),
    )
    parser.add_argument(
        "--policy-type",
        choices=sorted(POLICY_REGISTRY),
        default="smolvla",
        help=(
            "LeRobot policy class to load. 'smolvla' and 'act' ship with the base "
            "lerobot install; 'pi0' requires the [pi] extra."
        ),
    )
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--task",
        default="",
        help=(
            "Natural-language task string. Used by VLA models (smolvla, pi0); "
            "ignored by ACT."
        ),
    )
    parser.add_argument("--robot-type", default="franka")
    parser.add_argument("--obs-bind-ip", default="0.0.0.0")
    parser.add_argument("--obs-port", type=int, default=28081)
    parser.add_argument("--bridge-ip", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=28082)
    parser.add_argument("--rate-hz", type=float, default=30.0)
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
        help="Optional JSONL path for raw/clamped joint targets and clamp metadata.",
    )
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="Hold the current measured joint configuration without loading any policy; useful for bridge smoke tests.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=1.0,
        help=(
            "Exponential moving average coefficient for joint targets (0 < alpha <= 1.0). "
            "Lower values mean more smoothing; 1.0 disables EMA entirely (passthrough)."
        ),
    )
    parser.add_argument(
        "--tuning-port",
        type=int,
        default=None,
        help=(
            "When provided, start a UDP listener on this port for live parameter updates. "
            "Accepted keys: ema_alpha. "
            "Example: echo '{\"ema_alpha\": 0.3}' | nc -u -w1 127.0.0.1 28090"
        ),
    )
    parser.add_argument(
        "--butter-lowpass",
        action="store_true",
        help="Apply a butterworth lowpass filter to the joint actions.",
    )
    parser.add_argument(
        "--butter-lowpass-cutoff",
        type=float,
        default=1.0,
        help="Cutoff frequency for the butterworth lowpass filter.",
    )

    # --- RTC flags ---
    parser.add_argument(
        "--use-rtc",
        action="store_true",
        help=(
            "Enable Real-Time Chunking (RTC) for smooth inter-chunk transitions. "
            "Uses predict_action_chunk + ActionQueue instead of select_action. "
            "Compatible with flow-matching policies (smolvla, pi0)."
        ),
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=0,
        help=(
            "RTC: how many overlapping leftover steps to guide/blend against when inferring a new chunk. "
            "0 (default) auto-sets to the measured inference_delay for maximum "
            "reactivity. Higher values are smoother but less reactive. "
            "Only used when --use-rtc is set. (default: 0 = auto)"
        ),
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help=(
            "RTC: how strongly to enforce consistency with the previous chunk during "
            "flow-matching denoising. 10.0 is optimal for 10-step flow matching "
            "(SmolVLA, Pi0). Only used when --use-rtc is set. (default: 10.0)"
        ),
    )
    parser.add_argument(
        "--rtc-attention-schedule",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
        default="EXP",
        help=(
            "RTC: how guidance weight decays across the overlap region. "
            "EXP (exponential) is recommended. Only used when --use-rtc is set. "
            "(default: EXP)"
        ),
    )
    parser.add_argument(
        "--rtc-inference-delay",
        type=int,
        default=0,
        help=(
            "RTC: initial estimate for how many action steps the robot advances during one inference call. "
            "0 means auto-measure at startup; live RTC updates this from full chunk-production latency. "
            "Only used when --use-rtc is set. (default: 0 = auto)"
        ),
    )
    parser.add_argument(
        "--rtc-refill-threshold",
        type=int,
        default=0,
        help=(
            "RTC: request a new chunk when the queued action count is at or below this threshold. "
            "0 (default) auto-sets to current_inference_delay + execution_horizon so chunks overlap. "
            "Only used when --use-rtc is set. (default: 0 = auto)"
        ),
    )

    return parser.parse_args()

# Filter design
def design_butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Return numerator/denominator coefficients for the low-pass filter.
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Filter state initialization
def init_filter(b, a):
    zi = lfilter_zi(b, a)
    return zi

# Per-sample filtering
def process_sample(sample, b, a, zi):
    # lfilter returns filtered sample and new state
    filtered, new_zi = lfilter(b, a, [sample], zi=zi)
    return filtered[0], new_zi

def main() -> int:
    _ensure_supported_python()
    args = parse_args()
    if args.list_cameras:
        return _list_available_cameras()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be > 0")
    if not args.zero_actions and args.policy_path is None:
        raise ValueError("--policy-path is required unless --zero-actions is set")
    if not args.zero_actions and args.config is None:
        raise ValueError("--config is required unless --zero-actions is set")
    if args.camera_preview_samples <= 0:
        raise ValueError("--camera-preview-samples must be > 0")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1]")
    if args.rtc_refill_threshold < 0:
        raise ValueError("--rtc-refill-threshold must be >= 0")
    _validate_rtc_flags(args)
    if args.use_rtc and args.policy_type not in RTC_COMPATIBLE_POLICY_TYPES:
        raise ValueError(
            f"--use-rtc is not compatible with --policy-type {args.policy_type!r}. "
            f"RTC requires a flow-matching policy: {sorted(RTC_COMPATIBLE_POLICY_TYPES)}"
        )

    deployment_config: dict[str, Any] = {}
    if args.config is not None:
        deployment_config = _load_yaml_config(args.config)
        robot_cfg = deployment_config.get("robot", {}) or {}
        if "bind_ip" in robot_cfg:
            args.obs_bind_ip = str(robot_cfg["bind_ip"])
        if "port" in robot_cfg:
            args.obs_port = int(robot_cfg["port"])

    obs_rx = LatestRobotObservation(args.obs_bind_ip, args.obs_port)
    obs_rx.start()
    action_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.bridge_ip, args.action_port)
    period_s = 1.0 / args.rate_hz
    keyboard = KeyboardMonitor()

    tuner: LiveTuningReceiver | None = None
    if args.tuning_port is not None:
        tuner = LiveTuningReceiver(port=args.tuning_port, defaults={"ema_alpha": args.ema_alpha})
        tuner.start()
        print(f"Live tuning receiver active on udp://127.0.0.1:{args.tuning_port}", flush=True)
    ema_prev_joints: np.ndarray | None = None

    def get_ema_alpha() -> float:
        return tuner.get("ema_alpha") if tuner is not None else args.ema_alpha

    def log_event_marker(event_type: str, sequence_id: int, **fields: Any) -> None:
        latest_obs = obs_rx.latest()
        row = {
            "timestamp_ns": time.monotonic_ns(),
            "record_type": "event",
            "event_type": event_type,
            "sequence_id": sequence_id,
            "policy_type": None if args.zero_actions else args.policy_type,
            "robot_observation_timestamp_ns": None if latest_obs is None else latest_obs.get("timestamp_ns"),
            **fields,
        }
        _write_jsonl(action_log, row)

    policy = None
    preprocess = None
    postprocess = None
    policy_state_dim = None
    camera_sources: list[_CameraSource] = []
    action_log: TextIO | None = None
    preview_dir: Path | None = None
    action_queue: Any = None  # RTC ActionQueue when --use-rtc
    rtc_action_producer: RTCActionProducer | None = None
    rtc_inference_delay: int = 0
    rtc_refill_threshold: int | None = None

    gripper_hold_requirement = GripperHoldRequirement(command_count_threshold=7)
    joint_filters = [design_butter_lowpass(cutoff=args.butter_lowpass_cutoff, fs=30.0, order=3) for _ in range(JOINT_ACTION_DIM)]
    zi = None

    try:
        if args.log_actions_jsonl is not None:
            args.log_actions_jsonl.expanduser().parent.mkdir(parents=True, exist_ok=True)
            action_log = args.log_actions_jsonl.expanduser().open("a", buffering=1)

        if not args.zero_actions:
            _ensure_lerobot_importable(_resolve_lerobot_root(args.lerobot_root))
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference

            try:
                policy_class_loader = POLICY_REGISTRY[args.policy_type]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown --policy-type {args.policy_type!r}; choose from "
                    f"{sorted(POLICY_REGISTRY)}"
                ) from exc
            try:
                policy_class = policy_class_loader()
            except ImportError as exc:
                hint = ""
                if args.policy_type == "pi0":
                    hint = (
                        " Install Pi0 dependencies first, e.g. "
                        "'uv pip install -e \".[pi]\"' inside the lerobot venv."
                    )
                raise ImportError(
                    f"Failed to import policy class for --policy-type "
                    f"{args.policy_type!r}: {exc}.{hint}"
                ) from exc

            device = torch.device(args.device)
            policy_path = str(_resolve_policy_path(args.policy_path))

            # Configure RTC on the policy config before loading if requested.
            if args.use_rtc:
                from lerobot.configs import RTCAttentionSchedule
                from lerobot.policies.rtc.configuration_rtc import RTCConfig

                schedule_map = {
                    "EXP": RTCAttentionSchedule.EXP,
                    "LINEAR": RTCAttentionSchedule.LINEAR,
                    "ONES": RTCAttentionSchedule.ONES,
                    "ZEROS": RTCAttentionSchedule.ZEROS,
                }

                # Use a temporary execution_horizon; will be updated after
                # measuring inference delay if the user didn't set it explicitly.
                initial_execution_horizon = args.rtc_execution_horizon if args.rtc_execution_horizon > 0 else 10

                rtc_config = RTCConfig(
                    enabled=True,
                    execution_horizon=initial_execution_horizon,
                    max_guidance_weight=args.rtc_max_guidance_weight,
                    prefix_attention_schedule=schedule_map[args.rtc_attention_schedule],
                )

                # Load normally, then apply RTC config after loading since
                # from_pretrained loads config from the checkpoint.
                policy = policy_class.from_pretrained(policy_path)
                policy.config.rtc_config = rtc_config
                policy.init_rtc_processor()
            else:
                policy = policy_class.from_pretrained(policy_path)

            policy.to(device)
            policy.eval()
            image_keys = _obs_keys_from_config(deployment_config)
            policy_features = _validate_policy_features(policy, image_keys, args.print_policy_features)
            state_shape = _feature_shape(policy.config.input_features.get(OBS_STATE_KEY))
            if state_shape is None or len(state_shape) != 1:
                raise ValueError(f"Could not determine state dimension for {OBS_STATE_KEY!r}")
            policy_state_dim = int(state_shape[0])
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            fallback_hw = (args.camera_height, args.camera_width)
            camera_sources = _build_camera_sources(deployment_config, policy, fallback_hw)
            preview_dir = None if args.skip_preview_frames else _timestamped_preview_dir(args.preview_dir)
            if action_log is None and preview_dir is not None:
                default_log_path = preview_dir / "policy_actions.jsonl"
                default_log_path.parent.mkdir(parents=True, exist_ok=True)
                action_log = default_log_path.open("a", buffering=1)
            preview_start = time.monotonic()
            preview_images: dict[str, np.ndarray] = {}
            for _ in range(args.camera_preview_samples):
                preview_images = _read_images(camera_sources)
            preview_elapsed_s = time.monotonic() - preview_start
            camera_previews = []
            for source in camera_sources:
                props = source.camera.properties(source.zed_view) if source.zed_view is not None else source.camera.properties()
                summary = _preview_summary_from_image(
                    props,
                    source.obs_key,
                    source.target_hw,
                    args.camera_preview_samples,
                    preview_elapsed_s,
                    preview_images[source.obs_key],
                    preview_dir,
                )
                _print_camera_summary(summary)
                camera_previews.append(summary)
            if preview_dir is not None:
                manifest_path = preview_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "policy_type": args.policy_type,
                            "policy_path": policy_path,
                            "task": args.task,
                            "policy_features": policy_features,
                            "rtc_enabled": args.use_rtc,
                            "ema_smoothing": get_ema_alpha(),
                            "butter_lowpass": args.butter_lowpass,
                            "butter_lowpass_cutoff": args.butter_lowpass_cutoff,
                            "cameras": camera_previews,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"Saved camera preview manifest to {manifest_path}", flush=True)
            print(
                f"Loaded {args.policy_type} policy ({policy_class.__name__}) from {policy_path}",
                flush=True,
            )

            # RTC: initialize ActionQueue and measure inference delay.
            if args.use_rtc:
                from lerobot.policies.rtc.action_queue import ActionQueue

                # Wait for a robot observation before measuring inference delay.
                print("RTC: waiting for robot observation to measure inference delay...", flush=True)
                while obs_rx.latest() is None:
                    time.sleep(0.05)

                if args.rtc_inference_delay > 0:
                    rtc_inference_delay = args.rtc_inference_delay
                    print(f"RTC: using manually set inference_delay={rtc_inference_delay} steps", flush=True)
                else:
                    rtc_inference_delay = _measure_inference_delay(
                        policy=policy,
                        preprocess=preprocess,
                        postprocess=postprocess,
                        prepare_observation_for_inference=prepare_observation_for_inference,
                        camera_sources=camera_sources,
                        obs=obs_rx.latest(),
                        policy_state_dim=policy_state_dim,
                        device=device,
                        task=args.task,
                        robot_type=args.robot_type,
                        rate_hz=args.rate_hz,
                    )

                # Auto-derive execution_horizon from measured inference_delay
                # for maximum reactivity (re-observe as often as possible).
                if args.rtc_execution_horizon <= 0:
                    final_execution_horizon = rtc_inference_delay
                    print(
                        f"RTC: auto-set execution_horizon={final_execution_horizon} "
                        f"(= inference_delay) for maximum reactivity",
                        flush=True,
                    )
                else:
                    final_execution_horizon = args.rtc_execution_horizon
                    if final_execution_horizon < rtc_inference_delay:
                        print(
                            f"WARNING: --rtc-execution-horizon {final_execution_horizon} is less than "
                            f"measured inference_delay {rtc_inference_delay}. The action queue may "
                            f"drain before new chunks arrive, causing gaps.",
                            flush=True,
                        )

                # Update the RTCConfig with the final execution_horizon and
                # recreate the ActionQueue with the correct config.
                policy.config.rtc_config.execution_horizon = final_execution_horizon
                action_queue = ActionQueue(policy.config.rtc_config)
                rtc_refill_threshold = args.rtc_refill_threshold if args.rtc_refill_threshold > 0 else None

                rtc_action_producer = RTCActionProducer(
                    policy=policy,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    prepare_observation_for_inference=prepare_observation_for_inference,
                    camera_sources=camera_sources,
                    obs_rx=obs_rx,
                    action_queue=action_queue,
                    policy_state_dim=policy_state_dim,
                    device=device,
                    task=args.task,
                    robot_type=args.robot_type,
                    period_s=period_s,
                    initial_inference_delay=rtc_inference_delay,
                    execution_horizon=final_execution_horizon,
                    refill_threshold=rtc_refill_threshold,
                )
                rtc_action_producer.start()

                print(
                    f"RTC ready: execution_horizon={final_execution_horizon} "
                    f"initial_inference_delay={rtc_inference_delay} "
                    f"refill_threshold={rtc_refill_threshold or 'auto'} "
                    f"max_guidance_weight={args.rtc_max_guidance_weight} "
                    f"attention_schedule={args.rtc_attention_schedule}",
                    flush=True,
                )
        else:
            torch = None
            prepare_observation_for_inference = None

        print(
            f"Streaming policy joint targets to udp://{args.bridge_ip}:{args.action_port} "
            f"from observations udp://{args.obs_bind_ip}:{args.obs_port}"
            f"{' (RTC enabled)' if args.use_rtc else ''}",
            flush=True,
        )
        keyboard.start()
        if keyboard.enabled:
            print(
                "Operator controls: [p] pause policy, [h] pause and re-home the arm, "
                "[r] resume policy, [q] quit",
                flush=True,
            )
        else:
            print(
                "Operator key controls unavailable because stdin is not a TTY. "
                "The runner will stream automatically until interrupted.",
                flush=True,
            )

        sequence_id = 0
        operator_paused = False
        operator_rehome_pause = False
        operator_request_id = 0
        rehome_request_retries_remaining = 0
        while True:
            start = time.monotonic()
            for key in keyboard.poll():
                normalized = key.lower()
                if normalized == "p":
                    operator_paused = True
                    operator_rehome_pause = False
                    ema_prev_joints = None
                    if rtc_action_producer is not None:
                        rtc_action_producer.set_active(False, clear_queue=True, reason="operator_pause")
                    elif action_queue is not None:
                        action_queue.clear()
                    print("Policy paused by operator.", flush=True)
                elif normalized == "h":
                    operator_paused = True
                    ema_prev_joints = None
                    producer_idle = True
                    policy_reset_applied = False
                    if rtc_action_producer is not None:
                        rtc_action_producer.set_active(False, clear_queue=True, reason="operator_rehome")
                        producer_idle = rtc_action_producer.wait_until_idle(timeout_s=2.0)
                    elif action_queue is not None:
                        action_queue.clear()
                    if policy is not None and producer_idle:
                        policy.reset()
                        policy_reset_applied = True
                    elif policy is not None:
                        print("WARNING: RTC producer still busy; skipped policy.reset() for re-home.", flush=True)
                    operator_rehome_pause = True
                    operator_request_id += 1
                    rehome_request_retries_remaining = REHOME_REQUEST_REPEAT_PACKETS
                    gripper_hold_requirement.reset()
                    log_event_marker(
                        "operator_rehome_requested",
                        sequence_id,
                        operator_request_id=operator_request_id,
                        operator_paused=True,
                        operator_rehome_pause=True,
                        policy_reset=policy_reset_applied,
                    )
                    print(
                        f"Re-home requested by operator (request_id={operator_request_id}). "
                        "Policy will stay paused until you press 'r'.",
                        flush=True,
                    )
                elif normalized == "r":
                    operator_paused = False
                    operator_rehome_pause = False
                    log_event_marker(
                        "operator_resume_requested",
                        sequence_id,
                        operator_request_id=operator_request_id,
                        operator_paused=False,
                        operator_rehome_pause=False,
                    )
                    if rtc_action_producer is not None:
                        rtc_action_producer.set_active(True, clear_queue=True, reason="operator_resume")
                    print("Policy resume requested by operator.", flush=True)
                elif normalized == "q":
                    log_event_marker(
                        "operator_shutdown_requested",
                        sequence_id,
                        operator_request_id=operator_request_id,
                        operator_paused=operator_paused,
                        operator_rehome_pause=operator_rehome_pause,
                    )
                    print("Operator requested shutdown.", flush=True)
                    return 0

            obs = obs_rx.latest()
            if obs is None:
                time.sleep(min(period_s, 0.05))
                continue

            sequence_id += 1
            request_rehome = rehome_request_retries_remaining > 0
            enabled = (not operator_paused) and not request_rehome

            if args.zero_actions:
                raw_action = _current_hold_action(obs)
            elif operator_paused:
                hold_gripper_command = 0.0 if operator_rehome_pause else None
                raw_action = _current_hold_action(obs, gripper_command_override=hold_gripper_command)
            elif args.use_rtc:
                # ---- RTC path: consume queued chunks while a background thread infers ----
                assert action_queue is not None
                assert rtc_action_producer is not None

                rtc_action_producer.raise_if_failed()
                raw_action_tensor = action_queue.get()
                if raw_action_tensor is not None:
                    raw_action = raw_action_tensor.squeeze(0).detach().cpu().numpy()
                    if action_queue.qsize() <= rtc_action_producer.refill_threshold():
                        rtc_action_producer.request_inference("queue_low_after_pop")
                else:
                    # Queue is empty during startup or after an overrun; hold and wake the producer.
                    raw_action = _current_hold_action(obs)
                    rtc_action_producer.notify_queue_underrun()
            else:
                # ---- Sync path: select_action (existing behavior, unchanged) ----
                assert policy is not None
                assert preprocess is not None
                assert postprocess is not None
                assert policy_state_dim is not None
                assert torch is not None
                assert prepare_observation_for_inference is not None

                raw_observation = {
                    OBS_STATE_KEY: _robot_state_vector(obs, policy_state_dim),
                    **_read_images(camera_sources),
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

            action, gripper_command, clamp_info = _clamp_action_with_info(raw_action)
            gripper_hold_requirement.update(gripper_command) # Apply the hold requirement to the gripper command
            model_output_log = _model_output_log_fields(raw_action)
            
            alpha = get_ema_alpha()
            ema_applied = ema_prev_joints is not None and not operator_paused and alpha < 1.0
            if ema_applied:
                action[:JOINT_ACTION_DIM] = alpha * action[:JOINT_ACTION_DIM] + (1.0 - alpha) * ema_prev_joints
            ema_prev_joints = action[:JOINT_ACTION_DIM].copy()

            if args.butter_lowpass:
                if zi is None:
                    zi = [init_filter(*joint_filters[i])*action[i] for i in range(len(joint_filters))]

                for i in range(JOINT_ACTION_DIM):
                    action[i], zi[i] = process_sample(action[i], *joint_filters[i], zi[i])

            rtc_log_fields = (
                rtc_action_producer.snapshot()
                if args.use_rtc and rtc_action_producer is not None
                else {}
            )
            _write_jsonl(action_log, {
                "timestamp_ns": time.monotonic_ns(),
                "sequence_id": sequence_id,
                "source": "zero_actions" if args.zero_actions else ("rtc" if args.use_rtc else "policy"),
                "policy_type": None if args.zero_actions else args.policy_type,
                "task": args.task,
                "robot_observation_timestamp_ns": obs.get("timestamp_ns"),
                "enabled": enabled,
                "operator_paused": operator_paused,
                "operator_rehome_pause": operator_rehome_pause,
                "operator_request_id": operator_request_id,
                "request_rehome": request_rehome,
                "action_space": "joint_position_absolute",
                "gripper_command": gripper_command,
                "latched_gripper_command": gripper_hold_requirement.get(),
                "raw_joint_positions_rad": _jsonable_action(raw_action),
                "clamped_joint_positions_rad": _jsonable_action(action),
                "ema_alpha": alpha,
                "ema_applied": ema_applied,
                "butter_lowpass": args.butter_lowpass,
                "butter_lowpass_cutoff": args.butter_lowpass_cutoff,
                "rtc_enabled": args.use_rtc,
                "rtc_inference_delay": (
                    rtc_log_fields.get("rtc_current_inference_delay", rtc_inference_delay)
                    if args.use_rtc
                    else None
                ),
                **rtc_log_fields,
                **clamp_info,
                **model_output_log,
            })
            _send_action(
                action_sock,
                dst,
                sequence_id,
                action,
                gripper_hold_requirement.get(),
                enabled=enabled,
                operator_request_id=operator_request_id,
                request_rehome=request_rehome,
            )
            if rehome_request_retries_remaining > 0:
                rehome_request_retries_remaining -= 1

            elapsed = time.monotonic() - start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        return 0
    finally:
        keyboard.stop()
        if rtc_action_producer is not None:
            rtc_action_producer.stop()
        if tuner is not None:
            tuner.stop()
        obs_rx.stop()
        action_sock.close()
        if action_log is not None:
            action_log.close()
        seen_camera_ids: set[int] = set()
        for source in camera_sources:
            if id(source.camera) not in seen_camera_ids:
                seen_camera_ids.add(id(source.camera))
                source.camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
