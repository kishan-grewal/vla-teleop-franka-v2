"""Launch synchronized robot and camera data recorders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set

import yaml


STOP_REQUESTED = False


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_dir() -> Path:
    return script_dir().parent


def default_config_path() -> Path:
    return repo_dir() / "configs" / "data_collection.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the robot UDP recorder and configured camera recorders "
            "for one dataset collection session."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--recording-id",
        "--session-id",
        dest="recording_id",
        default="",
        help="Session folder name. Defaults to session_<YYYYmmdd_HHMMSS>.",
    )
    parser.add_argument("--recording-root", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop all recorders after seconds; 0 until Ctrl-C.")
    parser.add_argument("--robot-port", type=int, default=None)
    parser.add_argument("--disable-robot", action="store_true")
    parser.add_argument(
        "--disable-camera",
        action="append",
        default=[],
        metavar="CAMERA",
        help="Disable a configured camera by id or camera_name. Repeat as needed.",
    )
    parser.add_argument(
        "--enable-camera",
        action="append",
        default=[],
        metavar="CAMERA",
        help="Enable a configured camera by id or camera_name. Repeat as needed.",
    )
    parser.add_argument(
        "--camera-serial",
        action="append",
        default=[],
        metavar="CAMERA=SERIAL",
        help="Override a configured camera serial by id or camera_name. Repeat as needed.",
    )
    parser.add_argument(
        "--camera-depth",
        action="append",
        default=[],
        metavar="CAMERA",
        help="Force depth recording for a configured camera. Repeat as needed.",
    )
    parser.add_argument(
        "--camera-raw",
        action="append",
        default=[],
        metavar="CAMERA",
        help="Force backend raw recording for a camera: ZED .svo or RealSense .bag.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and session metadata without launching.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as src:
        config = yaml.safe_load(src) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML map")
    return config


def section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be a map")
    return value


def camera_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = config.get("cameras", [])
    if not isinstance(value, list):
        raise ValueError("config.cameras must be a list")

    cameras: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for index, raw_camera in enumerate(value):
        if not isinstance(raw_camera, dict):
            raise ValueError(f"config.cameras[{index}] must be a map")
        camera = dict(raw_camera)
        camera_id = str(camera.get("id") or camera.get("camera_name") or f"camera_{index}")
        if camera_id in seen_ids:
            raise ValueError(f"Duplicate camera id in config.cameras: {camera_id}")
        seen_ids.add(camera_id)
        camera["id"] = camera_id
        cameras.append(camera)
    return cameras


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_dir() / path


def bool_value(config: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(config.get(key, default))


def parse_key_value_overrides(values: List[str], flag: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for value in values:
        key, sep, override = value.partition("=")
        if not sep or not key:
            raise ValueError(f"{flag} expects CAMERA=VALUE; got {value!r}")
        overrides[key] = override
    return overrides


def camera_keys(cfg: Dict[str, Any]) -> Set[str]:
    keys = {str(cfg["id"])}
    for key in ("camera_name", "left_camera_name", "right_camera_name"):
        value = cfg.get(key)
        if value not in (None, ""):
            keys.add(str(value))
    return keys


def matches_camera(cfg: Dict[str, Any], names: Set[str]) -> bool:
    return bool(camera_keys(cfg) & names)


def camera_override_value(cfg: Dict[str, Any], overrides: Dict[str, str]) -> Optional[str]:
    for key in camera_keys(cfg):
        if key in overrides:
            return overrides[key]
    return None


def unknown_camera_names(cameras: List[Dict[str, Any]], names: Set[str]) -> List[str]:
    return sorted(name for name in names if not any(matches_camera(camera, {name}) for camera in cameras))


def add_flag(cmd: List[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def add_option(cmd: List[str], flag: str, value: Any, *, include_empty: bool = False) -> None:
    if value is None:
        return
    if value == "" and not include_empty:
        return
    cmd.extend([flag, str(value)])


def session_id_from_time() -> str:
    return "session_" + time.strftime("%Y%m%d_%H%M%S")


def write_session_metadata(
    session_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    config: Dict[str, Any],
    commands: Dict[str, List[str]],
) -> None:
    metadata = {
        "recording_id": session_dir.name,
        "session_dir": str(session_dir),
        "created_unix_time_ns": time.time_ns(),
        "config_path": str(config_path),
        "config": config,
        "commands": commands,
    }
    (session_dir / "session_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_robot_command(cfg: Dict[str, Any], session_dir: Path, duration_s: float, robot_port: Optional[int]) -> List[str]:
    cmd = [str(script_dir() / "record_robot_observations.py")]
    add_option(cmd, "--bind-ip", cfg.get("bind_ip", "0.0.0.0"))
    add_option(cmd, "--port", robot_port if robot_port is not None else cfg.get("port", 28081))
    add_option(cmd, "--output", session_dir / "robot.jsonl")
    add_option(cmd, "--episode-events-output", session_dir / "episode_events.jsonl")
    add_option(cmd, "--duration-s", duration_s)
    add_option(cmd, "--print-hz", cfg.get("print_hz", 1.0))
    add_option(cmd, "--flush-every", cfg.get("flush_every", 50))
    add_option(cmd, "--fsync-every", cfg.get("fsync_every", 0))
    add_flag(cmd, bool_value(cfg, "with_receive_metadata", False), "--with-receive-metadata")
    return cmd


def build_zed_command(
    cfg: Dict[str, Any],
    session_dir: Path,
    duration_s: float,
    serial_override: Optional[str],
    force_svo: bool,
    force_depth: bool,
) -> List[str]:
    if cfg.get("exposure") not in (None, -1) and bool_value(cfg, "auto_exposure", False):
        raise ValueError(f"config.cameras[{cfg['id']}].exposure and auto_exposure cannot both be set")
    cmd = [str(script_dir() / "record_zed_camera.py")]
    camera_name = cfg.get("camera_name", "ee_zed_m")
    left_camera_name = cfg.get("left_camera_name", f"{camera_name}_left")
    right_camera_name = cfg.get("right_camera_name", f"{camera_name}_right")
    add_option(cmd, "--camera-name", camera_name)
    add_option(cmd, "--output-dir", session_dir / "cameras" / str(camera_name))
    add_option(cmd, "--left-camera-name", left_camera_name)
    add_option(cmd, "--right-camera-name", right_camera_name)
    add_option(cmd, "--left-output-dir", session_dir / "cameras" / str(left_camera_name))
    add_option(cmd, "--right-output-dir", session_dir / "cameras" / str(right_camera_name))
    add_option(cmd, "--serial", serial_override if serial_override is not None else cfg.get("serial", 0))
    add_option(cmd, "--resolution", cfg.get("resolution", "HD720"))
    add_option(cmd, "--fps", cfg.get("fps", 30))
    add_option(cmd, "--width", cfg.get("width", 0))
    add_option(cmd, "--height", cfg.get("height", 0))
    add_option(cmd, "--duration-s", duration_s)
    add_option(cmd, "--svo-compression", cfg.get("svo_compression", "H264"))
    add_option(cmd, "--video-codec", cfg.get("video_codec", "mp4v"))
    add_option(cmd, "--print-hz", cfg.get("print_hz", 1.0))
    add_option(cmd, "--exposure", cfg.get("exposure"))
    add_flag(cmd, bool_value(cfg, "auto_exposure", False), "--auto-exposure")
    add_flag(cmd, bool_value(cfg, "svo", False) or force_svo, "--svo")
    add_flag(cmd, bool_value(cfg, "depth", False) or force_depth, "--depth")
    return cmd


def build_realsense_command(
    cfg: Dict[str, Any],
    session_dir: Path,
    duration_s: float,
    serial_override: Optional[str],
    force_bag: bool,
    force_depth: bool,
) -> List[str]:
    if cfg.get("exposure") not in (None, -1) and bool_value(cfg, "auto_exposure", False):
        raise ValueError(f"config.cameras[{cfg['id']}].exposure and auto_exposure cannot both be set")
    cmd = [str(script_dir() / "record_realsense_camera.py")]
    camera_name = cfg.get("camera_name", "third_person_d405")
    add_option(cmd, "--camera-name", camera_name)
    add_option(cmd, "--role", cfg.get("role", "third_person"))
    add_option(cmd, "--output-dir", session_dir / "cameras" / str(camera_name))
    add_option(cmd, "--serial", serial_override if serial_override is not None else cfg.get("serial", ""))
    add_option(cmd, "--color-width", cfg.get("color_width", 1280))
    add_option(cmd, "--color-height", cfg.get("color_height", 720))
    add_option(cmd, "--depth-width", cfg.get("depth_width", 640))
    add_option(cmd, "--depth-height", cfg.get("depth_height", 480))
    add_option(cmd, "--fps", cfg.get("fps", 30))
    add_option(cmd, "--duration-s", duration_s)
    add_option(cmd, "--video-codec", cfg.get("video_codec", "mp4v"))
    add_option(cmd, "--print-hz", cfg.get("print_hz", 1.0))
    add_option(cmd, "--exposure", cfg.get("exposure"))
    add_flag(cmd, bool_value(cfg, "auto_exposure", False), "--auto-exposure")
    add_flag(cmd, bool_value(cfg, "depth", False) or force_depth, "--depth")
    add_flag(cmd, not bool_value(cfg, "align_depth", True), "--no-align-depth")
    add_flag(cmd, bool_value(cfg, "bag", False) or force_bag, "--bag")
    return cmd


def build_camera_command(
    cfg: Dict[str, Any],
    session_dir: Path,
    duration_s: float,
    serial_overrides: Dict[str, str],
    depth_overrides: Set[str],
    raw_overrides: Set[str],
) -> List[str]:
    backend = str(cfg.get("backend", "")).lower()
    serial_override = camera_override_value(cfg, serial_overrides)
    force_depth = matches_camera(cfg, depth_overrides)
    force_raw = matches_camera(cfg, raw_overrides)

    if backend == "zed":
        return build_zed_command(cfg, session_dir, duration_s, serial_override, force_raw, force_depth)
    if backend == "realsense":
        return build_realsense_command(cfg, session_dir, duration_s, serial_override, force_raw, force_depth)
    raise ValueError(f"config.cameras[{cfg['id']}].backend must be 'zed' or 'realsense'; got {backend!r}")


def print_command(name: str, cmd: List[str]) -> None:
    print(f"[{name}] " + " ".join(shlex.quote(part) for part in cmd), flush=True)


def terminate_processes(processes: Dict[str, subprocess.Popen[Any]]) -> None:
    for name, proc in processes.items():
        if proc.poll() is None:
            print(f"Stopping {name}...", flush=True)
            proc.terminate()
    deadline = time.monotonic() + 5.0
    for proc in processes.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
    for name, proc in processes.items():
        print(f"{name} exited rc={proc.poll()}", flush=True)


def main() -> int:
    args = parse_args()
    if args.duration_s < 0.0:
        print("--duration-s must be >= 0", file=sys.stderr)
        return 2
    try:
        serial_overrides = parse_key_value_overrides(args.camera_serial, "--camera-serial")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = load_config(config_path)
    recording_root = resolve_repo_path(args.recording_root or Path(config.get("recording_root", "recordings")))
    recording_id = args.recording_id or session_id_from_time()
    session_dir = recording_root / recording_id

    commands: Dict[str, List[str]] = {}
    robot_cfg = section(config, "robot")
    cameras = camera_configs(config)
    disabled_cameras = set(args.disable_camera)
    enabled_cameras = set(args.enable_camera)
    depth_overrides = set(args.camera_depth)
    raw_overrides = set(args.camera_raw)
    unknown_names = (
        unknown_camera_names(cameras, disabled_cameras)
        + unknown_camera_names(cameras, enabled_cameras)
        + unknown_camera_names(cameras, depth_overrides)
        + unknown_camera_names(cameras, raw_overrides)
        + unknown_camera_names(cameras, set(serial_overrides))
    )
    if unknown_names:
        print("Unknown configured camera name(s): " + ", ".join(sorted(set(unknown_names))), file=sys.stderr)
        return 2

    if bool_value(robot_cfg, "enabled", True) and not args.disable_robot:
        commands["robot"] = build_robot_command(robot_cfg, session_dir, args.duration_s, args.robot_port)
    for camera in cameras:
        explicitly_enabled = matches_camera(camera, enabled_cameras)
        explicitly_disabled = matches_camera(camera, disabled_cameras)
        if explicitly_disabled and not explicitly_enabled:
            continue
        if not (bool_value(camera, "enabled", True) or explicitly_enabled):
            continue
        commands[f"camera:{camera['id']}"] = build_camera_command(
            camera,
            session_dir,
            args.duration_s,
            serial_overrides,
            depth_overrides,
            raw_overrides,
        )

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "cameras").mkdir(exist_ok=True)
    write_session_metadata(session_dir, args, config_path, config, commands)

    print(f"session_dir={session_dir}", flush=True)
    for name, cmd in commands.items():
        print_command(name, cmd)
    if args.dry_run:
        return 0
    if not commands:
        print("No recorders enabled.", file=sys.stderr)
        return 2

    processes: Dict[str, subprocess.Popen[Any]] = {}
    try:
        for name, cmd in commands.items():
            processes[name] = subprocess.Popen(cmd, cwd=str(repo_dir()))
            time.sleep(0.5)

        while not STOP_REQUESTED:
            exited = [(name, proc.poll()) for name, proc in processes.items() if proc.poll() is not None]
            failed = [(name, rc) for name, rc in exited if rc != 0]
            if failed:
                for name, rc in failed:
                    print(f"{name} exited unexpectedly rc={rc}", file=sys.stderr, flush=True)
                return 1
            if len(exited) == len(processes):
                return 0
            time.sleep(0.25)
    finally:
        terminate_processes(processes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
