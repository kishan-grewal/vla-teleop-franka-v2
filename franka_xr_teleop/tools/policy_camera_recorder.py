#!/usr/bin/env python3
"""Asynchronous recorder for camera frames used by policy deployment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any, Mapping, TextIO

import numpy as np


@dataclass(frozen=True)
class _FrameBatch:
    images: Mapping[str, np.ndarray]
    metadata: dict[str, Any]
    submitted_monotonic_ns: int
    submitted_perf_counter_ns: int


@dataclass
class _StreamWriter:
    obs_key: str
    stream_dir: Path
    video_path: Path
    frames_path: Path
    video_writer: Any
    frames_file: TextIO
    width: int
    height: int
    frame_index: int = 0


@dataclass
class _TimingStats:
    count: int = 0
    total_ms: float = 0.0
    last_ms: float | None = None
    max_ms: float = 0.0

    def add_ms(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.last_ms = elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def avg_ms(self) -> float | None:
        if self.count == 0:
            return None
        return self.total_ms / self.count


class PolicyCameraRecorder:
    """Encode policy camera frames on a background thread.

    submit() is intentionally non-blocking: if the bounded queue is full, the
    frame batch is dropped and policy deployment continues.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        fps: float,
        codec: str = "mp4v",
        queue_size: int = 4,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be > 0")
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        if len(codec) != 4:
            raise ValueError("codec must be a four-character OpenCV codec")

        self.output_dir = output_dir.expanduser()
        self.fps = float(fps)
        self.codec = codec
        self.queue_size = int(queue_size)
        self._queue: queue.Queue[_FrameBatch] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="PolicyCameraRecorder", daemon=True)
        self._streams: dict[str, _StreamWriter] = {}
        self._lock = threading.Lock()
        self._started = False
        self._disabled = False
        self._last_error: str | None = None
        self._submitted_batches = 0
        self._submitted_frames = 0
        self._written_batches = 0
        self._written_frames = 0
        self._dropped_batches = 0
        self._dropped_frames = 0
        self._timings: dict[str, _TimingStats] = {
            "start_api": _TimingStats(),
            "submit_api": _TimingStats(),
            "submit_drop_api": _TimingStats(),
            "snapshot_api": _TimingStats(),
            "stop_api": _TimingStats(),
            "queue_wait_batch": _TimingStats(),
            "write_batch": _TimingStats(),
            "stream_lookup_frame": _TimingStats(),
            "rgb_convert_frame": _TimingStats(),
            "video_write_frame": _TimingStats(),
            "jsonl_serialize_frame": _TimingStats(),
            "jsonl_write_frame": _TimingStats(),
            "frame_total": _TimingStats(),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(metadata or {})

    def start(self) -> None:
        start_ns = time.perf_counter_ns()
        try:
            with self._lock:
                if self._started:
                    return
                self._started = True
            self._thread.start()
        finally:
            self._record_timing("start_api", start_ns)

    def submit(self, images: Mapping[str, np.ndarray], metadata: Mapping[str, Any] | None = None) -> bool:
        start_ns = time.perf_counter_ns()
        if not images:
            self._record_timing("submit_api", start_ns)
            return True

        with self._lock:
            if self._disabled:
                self._record_timing_locked("submit_api", start_ns)
                return False

        batch = _FrameBatch(
            images=dict(images),
            metadata=dict(metadata or {}),
            submitted_monotonic_ns=time.monotonic_ns(),
            submitted_perf_counter_ns=time.perf_counter_ns(),
        )
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            with self._lock:
                self._dropped_batches += 1
                self._dropped_frames += len(batch.images)
                self._record_timing_locked("submit_api", start_ns)
                self._record_timing_locked("submit_drop_api", start_ns)
            return False

        with self._lock:
            self._submitted_batches += 1
            self._submitted_frames += len(batch.images)
            self._record_timing_locked("submit_api", start_ns)
        return True

    def snapshot(self) -> dict[str, Any]:
        start_ns = time.perf_counter_ns()
        with self._lock:
            snapshot = {
                "output_dir": str(self.output_dir),
                "queue_size": self.queue_size,
                "queued_batches": self._queue.qsize(),
                "submitted_batches": self._submitted_batches,
                "submitted_frames": self._submitted_frames,
                "written_batches": self._written_batches,
                "written_frames": self._written_frames,
                "dropped_batches": self._dropped_batches,
                "dropped_frames": self._dropped_frames,
                "disabled": self._disabled,
                "last_error": self._last_error,
            }
            snapshot.update(self._timing_fields_locked())
        with self._lock:
            self._record_timing_locked("snapshot_api", start_ns)
            snapshot.update(self._timing_fields_locked())
            snapshot["snapshot_api_current_ms"] = _elapsed_ms(start_ns)
        return snapshot

    def stop(self, timeout_s: float = 5.0) -> None:
        start_ns = time.perf_counter_ns()
        try:
            self._stop.set()
            if self._started:
                self._thread.join(timeout=timeout_s)
                if self._thread.is_alive():
                    print(
                        f"WARNING: policy camera recorder did not finish within {timeout_s:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            self._record_timing("stop_api", start_ns)

    def _write_manifest(self, metadata: Mapping[str, Any]) -> None:
        manifest = {
            "created_unix_time_ns": time.time_ns(),
            "fps": self.fps,
            "codec": self.codec,
            "queue_size": self.queue_size,
            "metadata": dict(metadata),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def _run(self) -> None:
        try:
            import cv2

            while not self._stop.is_set() or not self._queue.empty():
                try:
                    batch = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._record_timing_ns(
                        "queue_wait_batch",
                        time.perf_counter_ns() - batch.submitted_perf_counter_ns,
                    )
                    write_batch_start_ns = time.perf_counter_ns()
                    self._write_batch(cv2, batch)
                    self._record_timing("write_batch", write_batch_start_ns)
                    with self._lock:
                        self._written_batches += 1
                        self._written_frames += len(batch.images)
                finally:
                    self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._disabled = True
                self._last_error = repr(exc)
            print(f"WARNING: policy camera recorder disabled after error: {exc}", file=sys.stderr, flush=True)
        finally:
            self._close_streams()

    def _write_batch(self, cv2: Any, batch: _FrameBatch) -> None:
        metadata = dict(batch.metadata)
        camera_properties_by_key = metadata.pop("camera_properties", None)
        for obs_key, image in batch.images.items():
            frame_start_ns = time.perf_counter_ns()
            convert_start_ns = time.perf_counter_ns()
            frame = self._as_bgr_frame(cv2, image)
            convert_ms = _elapsed_ms(convert_start_ns)
            self._record_timing_ms("rgb_convert_frame", convert_ms)

            stream_start_ns = time.perf_counter_ns()
            stream = self._stream_for(cv2, obs_key, frame)
            stream_lookup_ms = _elapsed_ms(stream_start_ns)
            self._record_timing_ms("stream_lookup_frame", stream_lookup_ms)
            if frame.shape[1] != stream.width or frame.shape[0] != stream.height:
                raise RuntimeError(
                    f"Frame size for {obs_key!r} changed from "
                    f"{stream.width}x{stream.height} to {frame.shape[1]}x{frame.shape[0]}"
                )

            video_write_start_ns = time.perf_counter_ns()
            stream.video_writer.write(frame)
            video_write_ms = _elapsed_ms(video_write_start_ns)
            self._record_timing_ms("video_write_frame", video_write_ms)
            record = {
                "frame_index": stream.frame_index,
                "obs_key": obs_key,
                "submitted_monotonic_ns": batch.submitted_monotonic_ns,
                "written_monotonic_ns": time.monotonic_ns(),
                "timing_rgb_convert_ms": convert_ms,
                "timing_stream_lookup_ms": stream_lookup_ms,
                "timing_video_write_ms": video_write_ms,
                **metadata,
            }
            if isinstance(camera_properties_by_key, Mapping):
                stream_properties = camera_properties_by_key.get(obs_key)
                if stream_properties is not None:
                    record["camera_properties"] = stream_properties

            serialize_start_ns = time.perf_counter_ns()
            line = json.dumps(record, separators=(",", ":"), sort_keys=True, default=str) + "\n"
            serialize_ms = _elapsed_ms(serialize_start_ns)
            self._record_timing_ms("jsonl_serialize_frame", serialize_ms)

            jsonl_write_start_ns = time.perf_counter_ns()
            stream.frames_file.write(line)
            jsonl_write_ms = _elapsed_ms(jsonl_write_start_ns)
            self._record_timing_ms("jsonl_write_frame", jsonl_write_ms)
            self._record_timing("frame_total", frame_start_ns)
            stream.frame_index += 1

    def _stream_for(self, cv2: Any, obs_key: str, frame: np.ndarray) -> _StreamWriter:
        if obs_key in self._streams:
            return self._streams[obs_key]

        stream_dir = self.output_dir / self._safe_stream_dir_name(obs_key)
        stream_dir.mkdir(parents=True, exist_ok=True)
        video_path = stream_dir / "rgb.mp4"
        frames_path = stream_dir / "frames.jsonl"
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
        frames_file = frames_path.open("a", buffering=1, encoding="utf-8")

        stream = _StreamWriter(
            obs_key=obs_key,
            stream_dir=stream_dir,
            video_path=video_path,
            frames_path=frames_path,
            video_writer=video_writer,
            frames_file=frames_file,
            width=width,
            height=height,
        )
        self._streams[obs_key] = stream
        stream_metadata = {
            "obs_key": obs_key,
            "video_path": str(video_path),
            "frames_path": str(frames_path),
            "width": width,
            "height": height,
            "fps": self.fps,
            "codec": self.codec,
            "created_unix_time_ns": time.time_ns(),
        }
        (stream_dir / "metadata.json").write_text(
            json.dumps(stream_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return stream

    def _as_bgr_frame(self, cv2: Any, image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Expected HxW, HxWx1, HxWx3, or HxWx4 image, got shape {frame.shape}")
        return np.ascontiguousarray(frame)

    def _close_streams(self) -> None:
        for stream in self._streams.values():
            stream.video_writer.release()
            stream.frames_file.close()

    def _record_timing(self, name: str, start_ns: int) -> None:
        with self._lock:
            self._record_timing_locked(name, start_ns)

    def _record_timing_ns(self, name: str, elapsed_ns: int) -> None:
        self._record_timing_ms(name, elapsed_ns / 1_000_000.0)

    def _record_timing_ms(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._timings[name].add_ms(elapsed_ms)

    def _record_timing_locked(self, name: str, start_ns: int) -> None:
        self._timings[name].add_ms(_elapsed_ms(start_ns))

    def _timing_fields_locked(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, timing in self._timings.items():
            fields[f"{name}_count"] = timing.count
            fields[f"{name}_last_ms"] = timing.last_ms
            fields[f"{name}_avg_ms"] = timing.avg_ms
            fields[f"{name}_max_ms"] = timing.max_ms if timing.count else None
        return fields

    @staticmethod
    def _safe_stream_dir_name(obs_key: str) -> str:
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", obs_key).strip("._-") or "camera"
        readable = readable.replace(".", "_")[:80].strip("_") or "camera"
        digest = hashlib.sha1(obs_key.encode("utf-8")).hexdigest()[:8]
        return f"{readable}_{digest}"


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0
