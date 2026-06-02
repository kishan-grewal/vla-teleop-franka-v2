# franka_xr_teleop

Franka Panda teleoperation bridge using **XRoboToolkit PC Service SDK** as the only operator input source.

This module is the Franka-specific interface layer between XR-Robotics tracking/controller data and the libfranka 1 kHz servo loop. It is focused on safe teleoperation and command execution.

## Scope

- Receive operator state through XR-Robotics SDK callbacks (`PXREAInit` / `PXREADeviceStateJson`)
- Map XR controller pose into bounded joint-space targets
- Enforce deadman/clutch/timeout/workspace/rate-limit safety
- Execute bounded joint position commands through libfranka
- Publish executed action + robot observation for later recording

Non-goals:
- dataset export
- camera recording
- model training

## Repo Layout

```text
franka_xr_teleop/
  cpp/
    teleop_bridge/
      main.cpp
      xrobotics_source.cpp
      xrobotics_source.h
      teleop_mapper.cpp
      teleop_mapper.h
      teleop_state_machine.cpp
      teleop_state_machine.h
      franka_controller.cpp
      franka_controller.h
      safety.cpp
      safety.h
      gripper.cpp
      gripper.h
      observation_pub.cpp
      observation_pub.h
      common_types.h
      CMakeLists.txt
  configs/
    teleop.yaml
    robot.yaml
    safety.yaml
    xr_frame.yaml
  schemas/
    xr_command.schema.json
    robot_observation.schema.json
```

## Input Mapping (Controller-only v1)

The bridge maps the **right controller** fields from XR-Robotics state JSON to teleop semantics:
- `grip`: deadman / arm-enable trigger
- `trigger`: gripper command `[0,1]` mapped to binary open/close with hold-on-contact semantics
- `primaryButton` (A): episode start marker when `teleop.a_button_toggles_robot_control: false`
- `secondaryButton` (B): episode end marker in that same mode, otherwise legacy start/split marker
- `pose` (`x,y,z,qx,qy,qz,qw`): target position/orientation input

Important: `teleop.a_button_toggles_robot_control` in `configs/teleop.yaml` must
match the Quest app's main-menu `Switch w/ A Button` setting.

- If Quest `Switch w/ A Button` is `Off`, set
  `teleop.a_button_toggles_robot_control: false` so `A=start` and `B=end`.
- If Quest `Switch w/ A Button` is `On`, set
  `teleop.a_button_toggles_robot_control: true` so the bridge does not steal
  `A` from the Quest-side control toggle.

## Message Contract (Internal)

Incoming normalized command fields used by the bridge:
- `timestamp_ns`
- `sequence_id`
- `teleop_enabled`
- `clutch_pressed`
- `target_position_xyz`
- `target_orientation_xyzw`
- `gripper_command`

`timestamp_ns` is stamped on command receive using workstation monotonic time so timeout-to-hold logic is clock-safe.

Outgoing observation stream (UDP JSON):
- robot state: `q`, `q_cmd`, `dq`, TCP pose, gripper width, gripper state
- executed action: action space, joint target, gripper command
- status: control mode, teleop state, packet age, fault flags

## Teleop State Machine

States:
- `DISCONNECTED`
- `CONNECTED_IDLE`
- `TELEOP_ARMED`
- `TELEOP_ACTIVE`
- `FAULT`

Motion only occurs in `TELEOP_ACTIVE` when:
- XR stream is healthy (packet age within timeout)
- deadman is true
- robot is not in reflex/fault mode

## Prerequisites

- Ubuntu 22.04
- Same Franka setup as `franka-sanity-checks` (`libfranka 0.9.2`, server version 5)
- XRoboToolkit PC Service installed/running (for example `/opt/apps/roboticsservice/runService.sh`)

## Build

```bash
cd /home/radu/vla-teleop-franka-v2/franka_xr_teleop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DXROBOTICS_SERVICE_ROOT=/opt/apps/roboticsservice
cmake --build build -j"$(nproc)"
```

If SDK paths are non-default, pass `-DXROBOTICS_SDK_ROOT=/path/to/SDK`.

## Run

For dataset recording workflows with robot observations and cameras, see
[`DATA_COLLECTION.md`](DATA_COLLECTION.md).

Dry-run (SDK input only, no robot motion):

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge --dry-run
```

Live teleop:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081
```

SmolVLA policy control keeps the same robot execution path, but replaces XR
controller input with absolute joint targets plus a gripper command over UDP:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081 \
  --control-source policy \
  --policy-action-port 28082
```

In another terminal, run the policy sidecar:

```bash
./tools/run_vla_policy.py \
  --policy-path /home/kishan/industry-project/SmolVLA-Testing/outputs/001_smolvla \
  --obs-port 28081 \
  --bridge-ip 127.0.0.1 \
  --action-port 28082 \
  --top-camera 0 \
  --third-person-camera 1 \
  --task "your task instruction"
```

`--policy-path` may point either at the SmolVLA training output directory
(`outputs/<dataset>_smolvla`) or directly at a LeRobot checkpoint export such
as `outputs/<dataset>_smolvla/checkpoints/last/pretrained_model`.

When `run_vla_policy.py` is attached to a terminal, it also accepts live
operator keys:

- `p`: pause policy actions (bridge receives `enabled=false`)
- `h`: pause policy actions and send a one-shot arm re-home request to the bridge
- `r`: resume policy actions after the arm has finished re-homing
- `q`: stop the runner

For a bridge smoke test without loading a model or cameras:

```bash
./tools/run_vla_policy.py --zero-actions --obs-port 28081 --action-port 28082
```

Record one dataset session from the UDP robot stream plus the configured
cameras:

```bash
./tools/record_data_collection_session.py \
  --recording-id session_001
```

The teleop bridge should already be running with `--obs-port 28081` or the
matching port from `configs/data_collection.yaml`.

Or manually record only UDP robot observations:

```bash
./tools/record_robot_observations.py \
  --port 28081 \
  --output recordings/session_001/robot.jsonl
```

Add `--with-receive-metadata` to include host receive timestamps and source
address around each observation record. Pressing the Oculus right-controller B
button emits a one-shot `status.episode_end` marker in the UDP stream and then
triggers the full gripper-open plus arm rehome sequence; the recorder writes
those markers to `recordings/session_001/episode_events.jsonl` by default.

Record the end-effector ZED/ZED-M camera on the same host:

```bash
./tools/record_zed_camera.py \
  --camera-name ee_zed_m \
  --output-dir recordings/session_001/cameras/ee_zed_m \
  --svo
```

This writes two virtual camera directories, `ee_zed_m_left/` and
`ee_zed_m_right/`, each with its own `rgb.mp4`, `frames.jsonl`, and
`metadata.json`. If `--svo` is enabled, the shared stereo SVO is kept under
`ee_zed_m/raw.svo`. Use `--depth` if metric depth should also be saved for the
left virtual camera.

Align robot observations to camera frames after recording:

```bash
./tools/align_robot_camera_jsonl.py \
  --robot-jsonl recordings/session_001/robot.jsonl \
  --camera ee_zed_m_left recordings/session_001/cameras/ee_zed_m_left/frames.jsonl \
  --camera ee_zed_m_right recordings/session_001/cameras/ee_zed_m_right/frames.jsonl \
  --output recordings/session_001/aligned_samples.jsonl
```

Save the robot's current joint positions as the startup home pose:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --save-home
```

This reads the current measured joint state and updates `configs/teleop.yaml` at
`teleop.start_joint_positions_rad`. No motion is commanded in this mode.

Hold-only mode (data path active, no robot motion):

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081 \
  --no-motion
```

## Safety Notes

The 1 kHz callback remains RT-safe:
- no parsing/network I/O in callback
- no dynamic allocation in callback
- lock-free handoff of latest operator command

Safety shaping includes:
- packet timeout -> hold
- clutch recenter
- max translation/rotation speed limits
- per-step delta limits
- workspace clamping
- jump rejection
