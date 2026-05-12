#!/usr/bin/env python3
"""Offline smoke test for PolicyCameraRecorder.

Run from the repo root or tools directory:

    python3 franka_xr_teleop/tools/test_policy_camera_recorder.py

This intentionally does not use pytest. It uses fake NumPy frames only, so no
robot, policy checkpoint, RealSense, or ZED hardware is required.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from policy_camera_recorder import PolicyCameraRecorder


__test__ = False  # Keep pytest from treating this smoke script as a test module.


CAMERA_KEYS = ("observation.images.top", "observation.images.wrist_d405")


def fake_images(step: int) -> dict[str, np.ndarray]:
    top = np.zeros((32, 48, 3), dtype=np.uint8)
    top[:, :, 0] = (step * 40) % 255
    top[:, :, 1] = 40
    top[:, :, 2] = 180

    wrist = np.zeros((24, 36, 3), dtype=np.uint8)
    wrist[:, :, 0] = 20
    wrist[:, :, 1] = (step * 50) % 255
    wrist[:, :, 2] = 220
    return {
        CAMERA_KEYS[0]: top,
        CAMERA_KEYS[1]: wrist,
    }


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_recording_outputs(output_dir: Path, expected_frames_per_camera: int) -> None:
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), f"missing recorder manifest: {manifest_path}"

    stream_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert len(stream_dirs) == len(CAMERA_KEYS), f"expected {len(CAMERA_KEYS)} stream dirs, got {stream_dirs}"

    seen_keys: set[str] = set()
    for stream_dir in stream_dirs:
        video_path = stream_dir / "rgb.mp4"
        frames_path = stream_dir / "frames.jsonl"
        metadata_path = stream_dir / "metadata.json"
        assert video_path.exists(), f"missing video file: {video_path}"
        assert video_path.stat().st_size > 0, f"empty video file: {video_path}"
        assert frames_path.exists(), f"missing frames jsonl: {frames_path}"
        assert metadata_path.exists(), f"missing stream metadata: {metadata_path}"

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        obs_key = str(metadata["obs_key"])
        seen_keys.add(obs_key)
        assert obs_key in CAMERA_KEYS

        records = load_jsonl(frames_path)
        assert len(records) == expected_frames_per_camera, (
            f"{frames_path} expected {expected_frames_per_camera} records, got {len(records)}"
        )
        for index, record in enumerate(records):
            assert record["frame_index"] == index
            assert record["obs_key"] == obs_key
            assert record["source"] == "offline_smoke"
            assert record["sequence_id"] == index
            assert record["robot_observation_timestamp_ns"] == 10_000 + index
            assert record["camera_properties"]["serial"] == f"fake-{obs_key}"
            assert record["timing_rgb_convert_ms"] >= 0.0
            assert record["timing_stream_lookup_ms"] >= 0.0
            assert record["timing_video_write_ms"] >= 0.0

    assert seen_keys == set(CAMERA_KEYS)


def smoke_recording_writes_video_and_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="policy_camera_recorder_") as tmp:
        output_dir = Path(tmp) / "recording"
        recorder = PolicyCameraRecorder(
            output_dir,
            fps=10.0,
            codec="mp4v",
            queue_size=8,
            metadata={"test_name": "smoke_recording_writes_video_and_metadata"},
        )
        recorder.start()
        for step in range(3):
            accepted = recorder.submit(
                fake_images(step),
                metadata={
                    "source": "offline_smoke",
                    "sequence_id": step,
                    "robot_observation_timestamp_ns": 10_000 + step,
                    "camera_properties": {
                        key: {"serial": f"fake-{key}", "step": step}
                        for key in CAMERA_KEYS
                    },
                },
            )
            assert accepted, f"submit unexpectedly dropped step {step}"

        recorder.stop(timeout_s=2.0)
        stats = recorder.snapshot()
        assert stats["disabled"] is False, stats
        assert stats["last_error"] is None, stats
        assert stats["written_batches"] == 3, stats
        assert stats["written_frames"] == 6, stats
        assert stats["dropped_batches"] == 0, stats
        assert stats["dropped_frames"] == 0, stats
        assert stats["submit_api_count"] == 3, stats
        assert stats["submit_api_avg_ms"] is not None, stats
        assert stats["submit_api_max_ms"] >= 0.0, stats
        assert stats["write_batch_count"] == 3, stats
        assert stats["video_write_frame_count"] == 6, stats
        assert stats["video_write_frame_avg_ms"] is not None, stats
        assert stats["rgb_convert_frame_count"] == 6, stats
        assert stats["jsonl_write_frame_count"] == 6, stats
        assert stats["queue_wait_batch_count"] == 3, stats
        assert_recording_outputs(output_dir, expected_frames_per_camera=3)


def smoke_queue_overflow_drops_without_blocking() -> None:
    with tempfile.TemporaryDirectory(prefix="policy_camera_recorder_drop_") as tmp:
        recorder = PolicyCameraRecorder(Path(tmp) / "recording", fps=10.0, queue_size=1)

        first = recorder.submit(fake_images(0), metadata={"source": "drop_smoke", "sequence_id": 0})
        second = recorder.submit(fake_images(1), metadata={"source": "drop_smoke", "sequence_id": 1})

        stats = recorder.snapshot()
        assert first is True
        assert second is False
        assert stats["submitted_batches"] == 1, stats
        assert stats["submitted_frames"] == len(CAMERA_KEYS), stats
        assert stats["dropped_batches"] == 1, stats
        assert stats["dropped_frames"] == len(CAMERA_KEYS), stats
        assert stats["submit_api_count"] == 2, stats
        assert stats["submit_drop_api_count"] == 1, stats
        assert stats["submit_drop_api_last_ms"] is not None, stats


def main() -> int:
    smoke_recording_writes_video_and_metadata()
    smoke_queue_overflow_drops_without_blocking()
    print("PolicyCameraRecorder offline smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
