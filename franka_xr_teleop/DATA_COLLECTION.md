# Data Collection

This document describes the recommended recording layout for VLA-style dataset
collection with robot observations plus one or more cameras.

## Session Layout

Use one folder per episode/session:

```text
recordings/session_001/
  robot.jsonl
  aligned_samples.jsonl
  cameras/
    ee_zed_m/
      raw.svo          # optional shared ZED stereo asset
    ee_zed_m_left/
      rgb.mp4
      frames.jsonl
      metadata.json
      depth/           # optional, left-view depth only
    ee_zed_m_right/
      rgb.mp4
      frames.jsonl
      metadata.json
    third_person_d405/
      rgb.mp4
      frames.jsonl
      metadata.json
      raw.bag          # optional
      depth/           # optional
```

`ee_zed_m_left/rgb.mp4` and `ee_zed_m_right/rgb.mp4` are the two synchronized
views of the stereo pair. `cameras/ee_zed_m/raw.svo` is optional, but useful as
a ZED SDK backup for later re-extraction.

## Clock Sync

The robot UDP observation packet includes `timestamp_ns`, stamped on the host
with a monotonic clock in the robot control callback.

The ZED recorder logs each frame with `host_timestamp_ns`, also using the same
host monotonic clock. Since the ZED is plugged into the same host computer, these
timestamps can be matched directly for frame-level sync.

If cameras are later moved to other machines, use clock synchronization
PTP/NTP and record receive timestamps on every machine.

## Record A Full Session

For normal collection, use the launcher to start robot observations, the
end-effector ZED, and the third-person D405 together:

```bash
./tools/record_data_collection_session.py \
  --recording-id session_001
```

The teleop bridge should already be running with `--obs-port` set to the same
robot UDP port in `configs/data_collection.yaml`.

Defaults live in:

```text
configs/data_collection.yaml
```

Common overrides:

```bash
./tools/record_data_collection_session.py \
  --recording-id session_002 \
  --duration-s 120 \
  --robot-port 28081 \
  --zed-serial 0 \
  --realsense-serial 128422271175
```

Useful flags:

- `--dry-run`: print the commands without launching recorders.
- `--disable-zed`, `--disable-realsense`, `--disable-robot`: run a subset.
- `--zed-svo`, `--zed-depth`: force ZED raw/depth recording on.
- `--realsense-bag`, `--realsense-depth`: force D405 bag/depth recording on.

The launcher writes `session_metadata.json` containing the resolved commands and
config used for the session.

## Record Robot Observations Manually

Start the teleop bridge with UDP observations enabled:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081
```

In another terminal, record the UDP observations:

```bash
./tools/record_robot_observations.py \
  --port 28081 \
  --output recordings/session_001/robot.jsonl \
  --episode-events-output recordings/session_001/events.jsonl
```

When `teleop.a_button_toggles_robot_control: false`, press the Oculus
right-controller A button to mark episode start and B to mark episode end. The
bridge emits one-shot `status.episode_start` / `status.episode_end` markers in
the UDP observation stream, and the recorder writes corresponding rows to the
events output file.

If `teleop.a_button_toggles_robot_control: true`, the bridge leaves A alone and
falls back to the legacy behavior where B emits only `episode_start`.

Optional flags:

- `--duration-s 60`: stop after a fixed duration.
- `--max-packets N`: stop after a fixed number of valid packets.
- `--episode-events-output PATH`: choose where episode markers are written.
- `--with-receive-metadata`: wrap each observation with receiver timestamp and
UDP source address.
- `--fsync-every N`: periodically force data to disk for more conservative
recording.

## Record The End-Effector ZED/ZED-M

Record the wrist/end-effector camera:

```bash
./tools/record_zed_camera.py \
  --camera-name ee_zed_m \
  --output-dir recordings/session_001/cameras/ee_zed_m \
  --svo
```

Outputs:

- `../ee_zed_m_left/rgb.mp4`: compressed left-eye RGB video for playback and
  labeling.
- `../ee_zed_m_left/frames.jsonl`: one row per left-eye frame with
  `host_timestamp_ns`.
- `../ee_zed_m_left/metadata.json`: left virtual camera metadata and
  calibration fields.
- `../ee_zed_m_right/rgb.mp4`: compressed right-eye RGB video for playback and
  labeling.
- `../ee_zed_m_right/frames.jsonl`: one row per right-eye frame with the same
  synchronized timestamps and frame indices as the left view.
- `../ee_zed_m_right/metadata.json`: right virtual camera metadata and
  calibration fields.
- `raw.svo`: optional shared ZED raw recording under `cameras/ee_zed_m/` when
  `--svo` is enabled.
- `../ee_zed_m_left/depth/`: optional metric depth frames aligned to the left
  virtual camera when `--depth` is enabled.

Useful flags:

- `--serial SERIAL`: select a specific ZED camera in a multi-camera setup.
- `--resolution HD720`: choose `VGA`, `HD720`, `HD1080`, or `HD2K`.
- `--fps 30`: requested camera FPS.
- `--depth`: save per-frame metric depth as compressed NPZ files.
- `--duration-s 60`: stop after a fixed duration.
- `--max-frames N`: stop after a fixed number of frames.

For labeling/classification, use `ee_zed_m_left/rgb.mp4` and/or
`ee_zed_m_right/rgb.mp4`. Keep `cameras/ee_zed_m/raw.svo` in parallel when you
want the option to re-extract frames, depth, or rectified streams later.

## Record The Third-Person RealSense D405

List connected RealSense devices:

```bash
./tools/record_realsense_camera.py --list-devices
```

Record the third-person camera:

```bash
./tools/record_realsense_camera.py \
  --camera-name third_person_d405 \
  --serial <D405_SERIAL> \
  --output-dir recordings/session_001/cameras/third_person_d405
```

Outputs:

- `rgb.mp4`: compressed RGB video for playback and labeling.
- `frames.jsonl`: one row per frame with `host_timestamp_ns` and video frame
  index.
- `metadata.json`: RealSense stream profiles, intrinsics, depth scale, and
  device metadata.
- `raw.bag`: optional RealSense SDK recording when `--bag` is enabled.
- `depth/`: optional depth frames when `--depth` is enabled.

Useful flags:

- `--depth`: save per-frame `z16` depth as compressed NPZ files.
- `--bag`: keep a RealSense `.bag` backup for SDK playback/re-extraction.
- `--color-width 1280 --color-height 720 --fps 30`: set RGB stream mode.
- `--depth-width 640 --depth-height 480`: set depth stream mode.
- `--no-align-depth`: keep depth in its native frame instead of aligning it to
  color.

## Align Robot And Camera Data

After recording, build aligned training samples:

```bash
./tools/align_robot_camera_jsonl.py \
  --robot-jsonl recordings/session_001/robot.jsonl \
  --camera ee_zed_m_left recordings/session_001/cameras/ee_zed_m_left/frames.jsonl \
  --camera ee_zed_m_right recordings/session_001/cameras/ee_zed_m_right/frames.jsonl \
  --camera third_person_d405 recordings/session_001/cameras/third_person_d405/frames.jsonl \
  --output recordings/session_001/aligned_samples.jsonl
```

The aligner nearest-neighbor matches each robot `timestamp_ns` to each camera's
`host_timestamp_ns`.

Useful flags:

- `--max-dt-ms 35`: reject matches farther than this threshold.
- `--camera NAME PATH`: repeat this flag for additional cameras.
- `--allow-missing`: keep robot samples even when a camera match is missing.

Each output row contains:

```json
{
  "timestamp_ns": 123,
  "robot": {},
  "cameras": {
    "ee_zed_m_left": {
      "dt_ns": -5000000,
      "frame": {
        "rgb_video": "rgb.mp4",
        "rgb_video_frame": 42
      }
    },
    "ee_zed_m_right": {
      "dt_ns": -5000000,
      "frame": {
        "rgb_video": "rgb.mp4",
        "rgb_video_frame": 42
      }
    }
  }
}
```

`dt_ns` is `camera_timestamp_ns - robot_timestamp_ns`. Smaller absolute values
mean tighter sync.

## Calibration Notes

For the end-effector camera, fill in `T_ee_camera` in the camera
`metadata.json` once you have the wrist camera calibration. This transform is
important if later training or analysis needs spatial grounding between the
robot TCP pose and camera observations.

For multiple cameras, use stable names such as:

- `ee_zed_m_left`
- `ee_zed_m_right`
- `third_person_d405`
- `front_zed`
- `side_zed`
- `overhead`

Then pass each camera to the aligner:

```bash
./tools/align_robot_camera_jsonl.py \
  --robot-jsonl recordings/session_001/robot.jsonl \
  --camera ee_zed_m_left recordings/session_001/cameras/ee_zed_m_left/frames.jsonl \
  --camera ee_zed_m_right recordings/session_001/cameras/ee_zed_m_right/frames.jsonl \
  --camera third_person_d405 recordings/session_001/cameras/third_person_d405/frames.jsonl \
  --output recordings/session_001/aligned_samples.jsonl
```
