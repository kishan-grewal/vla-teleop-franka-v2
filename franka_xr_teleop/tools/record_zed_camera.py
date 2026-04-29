#!/usr/bin/env python3
"""Record a ZED camera stream with frame timestamps for dataset sync."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Optional, Tuple

STOP_REQUESTED = False
EXPOSURE_AUTO_SENTINEL = -1
EXPOSURE_MIN = 0
EXPOSURE_MAX = 100
POST_EXPOSURE_SETTLE_GRABS = 3


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("recordings") / f"zed_ee_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a ZED/ZED-M camera as two virtual cameras, one per stereo eye, "
            "each with its own rgb.mp4 and frames.jsonl timestamps. Optionally also "
            "records shared SVO and left-view depth arrays."
        )
    )
    exposure_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--camera-name", default="ee_zed_m")
    parser.add_argument("--left-camera-name", default="", help="Virtual camera name for VIEW.LEFT.")
    parser.add_argument("--right-camera-name", default="", help="Virtual camera name for VIEW.RIGHT.")
    parser.add_argument("--left-output-dir", type=Path, default=None, help="Output directory for VIEW.LEFT files.")
    parser.add_argument("--right-output-dir", type=Path, default=None, help="Output directory for VIEW.RIGHT files.")
    parser.add_argument("--serial", type=int, default=0, help="ZED serial number; 0 uses first camera.")
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["VGA", "HD720", "HD1080", "HD2K"],
        help="ZED camera resolution (default: HD720).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS requested from ZED SDK.")
    parser.add_argument("--width", type=int, default=0, help="Optional output width; 0 uses camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional output height; 0 uses camera height.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop after seconds; 0 records until Ctrl-C.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after frames; 0 records until Ctrl-C.")
    parser.add_argument("--svo", action="store_true", help="Also record a ZED SVO file for raw playback/extraction.")
    parser.add_argument(
        "--svo-compression",
        default="H264",
        choices=["H264", "H265", "LOSSLESS"],
        help="SVO compression mode (default: H264).",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Also save per-frame metric depth as compressed NPZ files. This is storage-heavy.",
    )
    exposure_group.add_argument(
        "--exposure",
        type=int,
        default=None,
        help=(
            "Exposure level in the ZED SDK's documented range [0, 100], or -1 "
            "to restore auto exposure. Manual exposure disables auto exposure "
            "while leaving gain unchanged."
        ),
    )
    exposure_group.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Explicitly enable the camera's automatic exposure mode.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="OpenCV fourcc for each virtual camera's rgb.mp4 (default: mp4v; try avc1 if your OpenCV supports it).",
    )
    parser.add_argument("--print-hz", type=float, default=1.0, help="Progress print rate; 0 disables.")
    return parser.parse_args()


def import_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import cv2
    except ImportError:
        print("Missing dependency: opencv-python / cv2", file=sys.stderr)
        raise
    try:
        import numpy as np
    except ImportError:
        print("Missing dependency: numpy", file=sys.stderr)
        raise
    try:
        import pyzed.sl as sl
    except ImportError:
        print("Missing dependency: ZED SDK Python module pyzed.sl", file=sys.stderr)
        raise
    return cv2, np, sl


def enum_value(container: Any, name: str) -> Any:
    return getattr(container, name)


def svo_compression_value(sl: Any, name: str) -> Any:
    if name != "LOSSLESS":
        return enum_value(sl.SVO_COMPRESSION_MODE, name)
    for candidate in ("H264_LOSSLESS", "H265_LOSSLESS", "LOSSLESS"):
        if hasattr(sl.SVO_COMPRESSION_MODE, candidate):
            return getattr(sl.SVO_COMPRESSION_MODE, candidate)
    raise AttributeError("ZED SDK does not expose a lossless SVO compression mode")


def timestamp_to_ns(timestamp: Any) -> Optional[int]:
    if timestamp is None:
        return None
    for method_name in ("get_nanoseconds", "get_milliseconds", "get_microseconds"):
        method = getattr(timestamp, method_name, None)
        if method is None:
            continue
        value = int(method())
        if method_name == "get_milliseconds":
            value *= 1_000_000
        elif method_name == "get_microseconds":
            value *= 1_000
        return value
    try:
        return int(timestamp)
    except (TypeError, ValueError):
        return None


def camera_intrinsics(camera: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key in ("fx", "fy", "cx", "cy", "disto"):
        if hasattr(camera, key):
            value = getattr(camera, key)
            if key == "disto":
                try:
                    value = list(value)
                except TypeError:
                    pass
            data[key] = value
    return data


def get_calibration_metadata(zed: Any) -> Dict[str, Any]:
    info = zed.get_camera_information()
    calibration = None
    if hasattr(info, "camera_configuration") and hasattr(info.camera_configuration, "calibration_parameters"):
        calibration = info.camera_configuration.calibration_parameters
    elif hasattr(info, "calibration_parameters"):
        calibration = info.calibration_parameters

    metadata: Dict[str, Any] = {
        "serial_number": getattr(info, "serial_number", None),
        "camera_model": str(getattr(info, "camera_model", "")),
    }
    if calibration is not None:
        left = getattr(calibration, "left_cam", None)
        right = getattr(calibration, "right_cam", None)
        if left is not None:
            metadata["left_intrinsics"] = camera_intrinsics(left)
        if right is not None:
            metadata["right_intrinsics"] = camera_intrinsics(right)
        if hasattr(calibration, "R"):
            metadata["left_to_right_R"] = list(getattr(calibration, "R"))
        if hasattr(calibration, "T"):
            metadata["left_to_right_T"] = list(getattr(calibration, "T"))
    return metadata


def open_zed(args: argparse.Namespace, sl: Any) -> Any:
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = enum_value(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE if args.depth else sl.DEPTH_MODE.NONE
    if args.serial:
        init.set_from_serial_number(args.serial)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")
    return zed


def get_camera_setting(zed: Any, sl: Any, setting: Any, label: str) -> int:
    read_err, value = zed.get_camera_settings(setting)
    if read_err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to read ZED {label}: {read_err}")
    return value


def configure_exposure(args: argparse.Namespace, zed: Any, sl: Any) -> Optional[int]:
    want_auto_exposure = args.auto_exposure or args.exposure == EXPOSURE_AUTO_SENTINEL
    if want_auto_exposure:
        err = zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable ZED auto exposure: {err}")
        err = zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, EXPOSURE_AUTO_SENTINEL)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to reset ZED exposure to auto: {err}")
        aec_agc = get_camera_setting(zed, sl, sl.VIDEO_SETTINGS.AEC_AGC, "AEC_AGC state")
        if aec_agc != 1:
            raise RuntimeError(f"Requested ZED auto exposure, but camera reported AEC_AGC={aec_agc}")
        exposure = get_camera_setting(zed, sl, sl.VIDEO_SETTINGS.EXPOSURE, "exposure after enabling auto mode")
        return exposure

    if args.exposure is None:
        return None

    if not EXPOSURE_MIN <= args.exposure <= EXPOSURE_MAX:
        raise ValueError(
            "--exposure must be -1 for auto exposure, or within the ZED SDK "
            f"documented manual range [{EXPOSURE_MIN}, {EXPOSURE_MAX}]; got {args.exposure}"
        )

    err = zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to disable ZED auto exposure before manual exposure set: {err}")
    err = zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, args.exposure)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to set ZED exposure to {args.exposure}: {err}")

    aec_agc = get_camera_setting(zed, sl, sl.VIDEO_SETTINGS.AEC_AGC, "AEC_AGC state")
    if aec_agc != 0:
        raise RuntimeError(
            f"Requested manual ZED exposure {args.exposure}, but camera reported AEC_AGC={aec_agc}"
        )
    exposure = get_camera_setting(zed, sl, sl.VIDEO_SETTINGS.EXPOSURE, "exposure after setting it")
    if exposure != args.exposure:
        raise RuntimeError(
            f"Requested ZED exposure {args.exposure}, but camera reported exposure {exposure}"
        )
    return exposure


def enable_svo(args: argparse.Namespace, zed: Any, sl: Any, svo_path: Path) -> None:
    compression = svo_compression_value(sl, args.svo_compression)
    params = sl.RecordingParameters(str(svo_path), compression)
    err = zed.enable_recording(params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to enable SVO recording: {err}")


def should_stop(start_mono: float, duration_s: float, frame_index: int, max_frames: int) -> bool:
    if STOP_REQUESTED:
        return True
    if duration_s > 0.0 and (time.monotonic() - start_mono) >= duration_s:
        return True
    if max_frames > 0 and frame_index >= max_frames:
        return True
    return False


def create_video_writer(cv2: Any, path: Path, codec: str, fps: int, size: Tuple[int, int]) -> Any:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec[:4]), float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def bgra_to_bgr(cv2: Any, bgra: Any, output_size: Tuple[int, int]) -> Any:
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if (bgr.shape[1], bgr.shape[0]) != output_size:
        bgr = cv2.resize(bgr, output_size, interpolation=cv2.INTER_AREA)
    return bgr


def default_side_camera_name(base_name: str, side: str) -> str:
    return f"{base_name}_{side}"


def default_side_output_dir(base_output_dir: Path, side: str) -> Path:
    return base_output_dir.with_name(f"{base_output_dir.name}_{side}")


def relative_path(path: Optional[Path], start: Path) -> Optional[str]:
    if path is None:
        return None
    return os.path.relpath(path, start)


def build_camera_metadata(
    *,
    camera_name: str,
    companion_camera_name: str,
    camera_view: str,
    svo_ref: Optional[str],
    depth_dir: Optional[str],
    resolution: str,
    fps: int,
    out_w: int,
    out_h: int,
    frame_w: int,
    frame_h: int,
    exposure_mode: str,
    exposure_value: Optional[int],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    notes = {
        "T_ee_camera": "Fill this with the calibrated camera pose in end-effector frame.",
        "timestamp_sync": "host_timestamp_ns uses the same host monotonic clock as robot timestamp_ns.",
    }
    if camera_view == "right" and depth_dir is None:
        notes["depth"] = (
            "Metric depth is stored only for the left virtual camera because the ZED SDK depth "
            "measure is aligned to the left image frame."
        )

    return {
        "camera_name": camera_name,
        "camera_view": camera_view,
        "stereo_companion_camera": companion_camera_name,
        "created_unix_time_ns": time.time_ns(),
        "host_clock": "time.monotonic_ns",
        "rgb_video": "rgb.mp4",
        "frames": "frames.jsonl",
        "svo": svo_ref,
        "depth_dir": depth_dir,
        "requested_resolution": resolution,
        "requested_fps": fps,
        "image_width": out_w,
        "image_height": out_h,
        "source_image_width": frame_w,
        "source_image_height": frame_h,
        "exposure_mode": exposure_mode,
        "exposure_value": exposure_value,
        "calibration": calibration,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 2
    if args.duration_s < 0.0 or args.max_frames < 0:
        print("--duration-s and --max-frames must be >= 0", file=sys.stderr)
        return 2
    if args.exposure is not None and args.exposure != EXPOSURE_AUTO_SENTINEL and not (
        EXPOSURE_MIN <= args.exposure <= EXPOSURE_MAX
    ):
        print(
            f"--exposure must be -1 for auto exposure, or within the ZED SDK documented "
            f"manual range [{EXPOSURE_MIN}, {EXPOSURE_MAX}]",
            file=sys.stderr,
        )
        return 2
    left_camera_name = args.left_camera_name or default_side_camera_name(args.camera_name, "left")
    right_camera_name = args.right_camera_name or default_side_camera_name(args.camera_name, "right")
    if left_camera_name == right_camera_name:
        print("Left and right virtual camera names must be different.", file=sys.stderr)
        return 2

    left_output_dir = args.left_output_dir or default_side_output_dir(args.output_dir, "left")
    right_output_dir = args.right_output_dir or default_side_output_dir(args.output_dir, "right")
    if left_output_dir == right_output_dir:
        print("Left and right output directories must be different.", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    cv2, np, sl = import_dependencies()

    left_output_dir.mkdir(parents=True, exist_ok=True)
    right_output_dir.mkdir(parents=True, exist_ok=True)
    if args.svo:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_left_path = left_output_dir / "rgb.mp4"
    rgb_right_path = right_output_dir / "rgb.mp4"
    left_frames_path = left_output_dir / "frames.jsonl"
    right_frames_path = right_output_dir / "frames.jsonl"
    left_metadata_path = left_output_dir / "metadata.json"
    right_metadata_path = right_output_dir / "metadata.json"
    svo_path = args.output_dir / "raw.svo"
    depth_dir = left_output_dir / "depth"
    if args.depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    zed = open_zed(args, sl)
    if args.svo:
        enable_svo(args, zed, sl, svo_path)

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()

    # Grab one frame before creating the writer so the output dimensions match
    # the actual ZED image stream.
    err = zed.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Initial ZED grab failed: {err}")
    applied_exposure = configure_exposure(args, zed, sl)
    for _ in range(POST_EXPOSURE_SETTLE_GRABS):
        err = zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(f"ZED grab failed while settling exposure change: {err}")
    zed.retrieve_image(left_image, sl.VIEW.LEFT)
    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
    first_left_bgra = left_image.get_data()
    first_right_bgra = right_image.get_data()
    frame_h, frame_w = first_left_bgra.shape[:2]
    right_frame_h, right_frame_w = first_right_bgra.shape[:2]
    if (right_frame_w, right_frame_h) != (frame_w, frame_h):
        zed.close()
        raise RuntimeError(
            "Left/right ZED views have different source sizes: "
            f"left={frame_w}x{frame_h} right={right_frame_w}x{right_frame_h}"
        )
    out_w = args.width if args.width > 0 else frame_w
    out_h = args.height if args.height > 0 else frame_h
    output_size = (out_w, out_h)

    try:
        left_writer = create_video_writer(cv2, rgb_left_path, args.video_codec, args.fps, output_size)
        right_writer = create_video_writer(cv2, rgb_right_path, args.video_codec, args.fps, output_size)
    except Exception:
        zed.close()
        raise

    calibration = get_calibration_metadata(zed)
    left_svo_ref = relative_path(svo_path if args.svo else None, left_output_dir)
    right_svo_ref = relative_path(svo_path if args.svo else None, right_output_dir)
    left_metadata = build_camera_metadata(
        camera_name=left_camera_name,
        companion_camera_name=right_camera_name,
        camera_view="left",
        svo_ref=left_svo_ref,
        depth_dir="depth" if args.depth else None,
        resolution=args.resolution,
        fps=args.fps,
        out_w=out_w,
        out_h=out_h,
        frame_w=frame_w,
        frame_h=frame_h,
        exposure_mode=(
            "auto"
            if args.auto_exposure or args.exposure == EXPOSURE_AUTO_SENTINEL
            else "manual"
            if args.exposure is not None
            else "default"
        ),
        exposure_value=applied_exposure,
        calibration=calibration,
    )
    right_metadata = build_camera_metadata(
        camera_name=right_camera_name,
        companion_camera_name=left_camera_name,
        camera_view="right",
        svo_ref=right_svo_ref,
        depth_dir=None,
        resolution=args.resolution,
        fps=args.fps,
        out_w=out_w,
        out_h=out_h,
        frame_w=frame_w,
        frame_h=frame_h,
        exposure_mode=(
            "auto"
            if args.auto_exposure or args.exposure == EXPOSURE_AUTO_SENTINEL
            else "manual"
            if args.exposure is not None
            else "default"
        ),
        exposure_value=applied_exposure,
        calibration=calibration,
    )
    left_metadata_path.write_text(json.dumps(left_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    right_metadata_path.write_text(json.dumps(right_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"Recording ZED {args.camera_name}: "
        f"{left_camera_name}={left_output_dir} {right_camera_name}={right_output_dir} "
        f"svo={int(args.svo)} depth={int(args.depth)} "
        f"exposure_mode={'auto' if args.auto_exposure or args.exposure == EXPOSURE_AUTO_SENTINEL else 'manual' if args.exposure is not None else 'default'} "
        f"exposure_value={applied_exposure}",
        flush=True,
    )

    frame_index = 0
    start_mono = time.monotonic()
    last_print = start_mono
    progress_period_s = 1.0 / args.print_hz if args.print_hz > 0.0 else 0.0

    def write_frame(
        left_bgra: Any,
        right_bgra: Any,
        zed_timestamp_ns: Optional[int],
        left_frames_file: Any,
        right_frames_file: Any,
    ) -> None:
        nonlocal frame_index
        host_timestamp_ns = time.monotonic_ns()
        left_writer.write(bgra_to_bgr(cv2, left_bgra, output_size))
        right_writer.write(bgra_to_bgr(cv2, right_bgra, output_size))

        left_depth_file: Optional[str] = None
        if args.depth:
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_arr = depth.get_data().astype(np.float32, copy=True)
            left_depth_file = f"depth_{frame_index:06d}.npz"
            np.savez_compressed(depth_dir / left_depth_file, depth_m=depth_arr)

        left_record = {
            "camera": left_camera_name,
            "frame_index": frame_index,
            "host_timestamp_ns": host_timestamp_ns,
            "zed_timestamp_ns": zed_timestamp_ns,
            "rgb_video": "rgb.mp4",
            "rgb_video_frame": frame_index,
            "depth_file": left_depth_file,
            "width": out_w,
            "height": out_h,
        }
        right_record = {
            "camera": right_camera_name,
            "frame_index": frame_index,
            "host_timestamp_ns": host_timestamp_ns,
            "zed_timestamp_ns": zed_timestamp_ns,
            "rgb_video": "rgb.mp4",
            "rgb_video_frame": frame_index,
            "depth_file": None,
            "width": out_w,
            "height": out_h,
        }
        left_frames_file.write(json.dumps(left_record, separators=(",", ":"), sort_keys=True) + "\n")
        right_frames_file.write(json.dumps(right_record, separators=(",", ":"), sort_keys=True) + "\n")
        frame_index += 1

    with left_frames_path.open("w", encoding="utf-8", buffering=1) as left_frames_file, \
        right_frames_path.open("w", encoding="utf-8", buffering=1) as right_frames_file:
        ts = timestamp_to_ns(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE))
        write_frame(first_left_bgra, first_right_bgra, ts, left_frames_file, right_frames_file)

        while not should_stop(start_mono, args.duration_s, frame_index, args.max_frames):
            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            ts = timestamp_to_ns(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE))
            write_frame(left_image.get_data(), right_image.get_data(), ts, left_frames_file, right_frames_file)

            now = time.monotonic()
            if progress_period_s > 0.0 and (now - last_print) >= progress_period_s:
                elapsed = max(now - start_mono, 1e-9)
                print(
                    f"frames={frame_index} rate={frame_index / elapsed:.1f}Hz "
                    f"left={left_output_dir} right={right_output_dir}"
                )
                last_print = now

    left_writer.release()
    right_writer.release()
    if args.svo:
        zed.disable_recording()
    zed.close()
    print(
        f"Stopped. frames={frame_index} left={left_output_dir} right={right_output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
