#!/usr/bin/env python3
"""Run a LeRobot policy (SmolVLA, ACT, or Pi0) and stream absolute 7-joint targets to the Franka bridge."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import select
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import numpy as np

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
# policies here (e.g. pi0fast, pi05) and they Just Work end-to-end.
POLICY_REGISTRY: dict[str, Callable[[], Any]] = {
    "smolvla": _load_smolvla_policy_class,
    "act": _load_act_policy_class,
    "pi0": _load_pi0_policy_class,
}


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
    return parser.parse_args()


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
        else:
            torch = None
            prepare_observation_for_inference = None

        print(
            f"Streaming policy joint targets to udp://{args.bridge_ip}:{args.action_port} "
            f"from observations udp://{args.obs_bind_ip}:{args.obs_port}",
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
                    print("Policy paused by operator.", flush=True)
                elif normalized == "h":
                    operator_paused = True
                    ema_prev_joints = None
                    if policy is not None:
                        policy.reset()
                    operator_rehome_pause = True
                    operator_request_id += 1
                    rehome_request_retries_remaining = REHOME_REQUEST_REPEAT_PACKETS
                    log_event_marker(
                        "operator_rehome_requested",
                        sequence_id,
                        operator_request_id=operator_request_id,
                        operator_paused=True,
                        operator_rehome_pause=True,
                        policy_reset=policy is not None,
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
            else:
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
            model_output_log = _model_output_log_fields(raw_action)
            alpha = get_ema_alpha()
            ema_applied = ema_prev_joints is not None and not operator_paused and alpha < 1.0
            if ema_applied:
                action[:JOINT_ACTION_DIM] = alpha * action[:JOINT_ACTION_DIM] + (1.0 - alpha) * ema_prev_joints
            ema_prev_joints = action[:JOINT_ACTION_DIM].copy()
            _write_jsonl(action_log, {
                "timestamp_ns": time.monotonic_ns(),
                "sequence_id": sequence_id,
                "source": "zero_actions" if args.zero_actions else "policy",
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
                "raw_joint_positions_rad": _jsonable_action(raw_action),
                "clamped_joint_positions_rad": _jsonable_action(action),
                "ema_alpha": alpha,
                "ema_applied": ema_applied,
                **model_output_log,
                **clamp_info,
            })
            _send_action(
                action_sock,
                dst,
                sequence_id,
                action,
                gripper_command,
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
