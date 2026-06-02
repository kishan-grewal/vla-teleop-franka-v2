# Real Robot Quick Start (Build + Run)

## 0) Preconditions

- Franka FCI is enabled in Desk.
- PC is on PREEMPT_RT kernel (for example `uname -r` contains `rt`).
- Quest USB/ADB path is working (see [QUEST3_CONNECTION.md](/home/radu/vla-teleop-franka-v2/QUEST3_CONNECTION.md)).
- XRoboToolkit PC service is installed at `/opt/apps/roboticsservice`.

## 1) Build bridge

```bash
cd /home/radu/vla-teleop-franka-v2/franka_xr_teleop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DXROBOTICS_SERVICE_ROOT=/opt/apps/roboticsservice
cmake --build build -j"$(nproc)"
```

## 2) Start Quest + XR service path

1. Verify Quest is visible:

```bash
adb devices -l
```

2. Set ADB reverse tunnel:

```bash
adb reverse --remove-all
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

3. In Quest app, connect to `127.0.0.1`.

4. Start XRoboToolkit PC service:

```bash
/opt/apps/roboticsservice/run3D.sh
```

## 3) Run modes

Dry-run (input path only, no robot motion):

```bash
cd /home/radu/vla-teleop-franka-v2/franka_xr_teleop
./build/cpp/teleop_bridge/franka_xr_teleop_bridge --dry-run
```

Hold-only (robot connected, no arm motion):

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081 \
  --no-motion
```

Live teleop:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge \
  --robot-ip 192.168.2.200 \
  --obs-port 28081
```

If robot is in `Reflex` mode, clear it first (no motion command):

```bash
cd /home/radu/vla-teleop-franka-v2/franka-sanity-checks
./build/panda_libfranka_sanity --robot-ip 192.168.2.200 --mode recover-only
```

Then start the teleop bridge again.

## 4) If you see realtime scheduling error

Symptom:

```text
Franka exception: libfranka: unable to set realtime scheduling: Operation not permitted
```

Immediate fix (recommended):

```bash
cd /home/radu/vla-teleop-franka-v2/franka_xr_teleop
sudo setcap cap_sys_nice,cap_ipc_lock+ep ./build/cpp/teleop_bridge/franka_xr_teleop_bridge
getcap ./build/cpp/teleop_bridge/franka_xr_teleop_bridge
```

Expected capability output includes:

```text
cap_sys_nice,cap_ipc_lock=ep
```

Then run bridge again.

Persistent user-side RT setup:

```bash
sudo groupadd -f realtime
sudo usermod -aG realtime $USER
```

Log out/in (or reboot), then verify:

```bash
ulimit -r
ulimit -l
```

`ulimit -r` should be non-zero (typically `99`).

Important:
- Rebuilding the binary removes `setcap`; run the `setcap` command again after each rebuild.

---

# Architecture Overview

You now have a 4-stage pipeline:

1. Quest input
- Quest app sends right-controller pose and buttons through XRoboToolkit PC Service.
- We use `right_grip` as the “move enabled” trigger and `right_trigger` for gripper.

2. Intent builder (safe target generation)
- When you first squeeze grip, the system “anchors” current robot pose and current controller pose.
- After that, hand motion is interpreted as relative motion from that anchor (not absolute world jumps).
- It applies the XR-to-robot frame mapping and scale factor.
- If grip is released, anchoring resets so re-engage starts from the robot’s current pose.

3. Planner (non-realtime)
- Runs in a normal thread (~100 Hz).
- Takes latest XR command + latest robot state.
- Produces a small next joint target using IK-like math with safety shaping:
  - position-only or full pose mode
  - singularity-aware damping
  - nullspace bias toward home posture
  - workspace/jump/timeout checks
- Publishes the newest joint target + status flags.

4. Robot servo loop (hard realtime)
- Runs at robot control rate (1 kHz).
- Reads latest planner target and sends smooth `JointPositions` commands.
- If target is stale, invalid, or motion disabled, it holds position.
- Observation stream publishes current state, mode, teleop state, and safety flags.

Startup and gripper:
- On startup it connects robot, tries gripper, recovers reflex if needed, moves to configured home pose, then starts teleop.
- Gripper runs in its own non-realtime loop: maps trigger to width, rate-limits commands, and avoids tiny command spam.

That’s the main design: Quest expresses operator intent, planner turns intent into safe incremental robot goals, realtime loop executes smoothly and safely.

---

## Two Threads System Design

**Planner Thread (`PlannerLoop`)**
Core loop is in [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:266).

1. Runs at `planner_rate_hz` (default 100 Hz), sleeping each cycle.
2. Reads latest XR input from `command_buffer` and latest robot snapshot from `robot_state_buffer`.
3. Builds a default `PlannedTarget` in HOLD mode (safe fallback).
4. Computes packet age and teleop state (`DISCONNECTED/IDLE/ARMED/ACTIVE`) using state machine logic.
5. If teleop is not active, it resets mapper/smoother and publishes HOLD target.
6. If active, it maps XR delta pose to desired robot TCP pose via anchor-based mapping.
7. Runs safety filter (timeout, workspace clamp, jump reject).
8. Solves IK to get joint target `q_target`.
9. Applies the joint smoother (your new part), then publishes final `PlannedTarget`.

**1 kHz Robot Callback (`robot.control` lambda)**
Execution loop is in [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:528).

1. Runs at robot servo rate (1 kHz, driven by `libfranka`).
2. Converts current robot state to `RobotSnapshot` and publishes to `robot_state_buffer`.
3. Reads latest `PlannedTarget` from `planned_target_buffer`.
4. Decides if motion should be applied:
`allow_motion && teleop_active && target_fresh && control_mode != HOLD`.
5. If applying motion, starts from `state.q_d` and steps toward planned target with per-cycle clamping.
6. Publishes `RobotObservation` (including `packet_age_ns`, `target_age_ns`, faults, etc.).
7. Returns `franka::JointPositions` for this cycle.

**Interaction Between Them**
Shared lock-free buffers are defined in [common_types.h](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/common_types.h:188).

1. Callback -> Planner path:
`robot_state_buffer` carries freshest robot state every 1 ms.
2. Planner -> Callback path:
`planned_target_buffer` carries newest planned joint target every ~10 ms.
3. They are decoupled by design:
planner does heavier logic/IK/safety at low rate; callback stays lightweight/deterministic.
4. No mutex handoff in hot path:
each side always reads the latest published value (double-buffer swap via atomics).
5. Practical effect:
the callback usually executes the same planned target for multiple 1 kHz ticks until planner publishes a new one.
6. Safety behavior coupling:
when planner decides HOLD (inactive, timeout, jump reject, IK fail), callback sees that and stops applying motion on next cycles.

That split is the key architecture: planner decides *what* target is safe and smooth, 1 kHz callback decides *how* to execute it stably at servo rate.

---

Great question. In your code, buffering is a **lock-free latest-value handoff**, not a FIFO queue.

**How buffering works**
1. Both buffers use the same template: [common_types.h](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/common_types.h:188).
2. Internally it has `slots_[2]` and an atomic `published_index_`.
3. `Publish(value)` writes to the other slot, then atomically flips index with `memory_order_release`.
4. `ReadLatest()` atomically reads index with `memory_order_acquire` and returns a copy of that slot.
5. Result: no locks, no partial reads, and old samples are overwritten. You always get the newest sample.

So:
- `command_buffer` = latest XR command.
- `robot_state_buffer` = latest Franka state snapshot.

**How Quest controller position becomes planner input**
1. Quest -> XRoboToolkit PC service -> SDK callback `PXREADeviceStateJson`.
   - Callback handling in [xrobotics_source.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/xrobotics_source.cpp:113).
2. JSON field `Controller.right.pose` is parsed from string `"x,y,z,qx,qy,qz,qw"` into `XRCommand.right_controller_pose`.
   - Parse logic: [xrobotics_source.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/xrobotics_source.cpp:22).
3. On receive, timestamp is set to local monotonic time (`cmd.timestamp_ns = MonotonicNowNs()`), then published to `command_buffer`.
   - [xrobotics_source.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/xrobotics_source.cpp:177)
   - [xrobotics_source.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/xrobotics_source.cpp:188)
4. Robot state is produced in the 1 kHz libfranka callback (`ToSnapshot(state)`) and published to `robot_state_buffer`.
   - [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:528)
   - [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:531)
5. Planner thread reads both latest values each cycle.
   - [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:284)
   - [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:285)

**Pose/frame transformation in planner path**
1. On first activation, planner stores anchor poses: current robot TCP + current XR controller pose.
   - [teleop_mapper.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_mapper.cpp:36)
2. Each cycle: `xr_delta = xr_current_pos - xr_anchor_pos`.
   - [teleop_mapper.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_mapper.cpp:43)
3. Delta is rotated into robot frame with `xr_to_robot_rotation` and scaled.
   - [teleop_mapper.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_mapper.cpp:48)
   - Matrix source: [xr_frame.yaml](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/configs/xr_frame.yaml:3)
4. Orientation uses quaternion delta (`anchor -> current`), converted to angle-axis, rotated to robot frame, then applied to anchor robot orientation.
   - [teleop_mapper.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_mapper.cpp:68)
5. That mapped pose then goes through safety + IK + smoothing to produce joint target.

If you want, I can also draw this as a compact signal-flow diagram (`Quest -> callback -> buffer -> mapper -> safety -> IK -> smoother -> 1kHz callback`).

---

## Is Franka planner implementation similar to XRoboToolkit implementations for other robots?

**Short answer**
Partially yes. Your planner thread follows the same **high-level teleop logic** as the official Python sample, but the architecture and safety/control rigor are significantly different.

**Where it is similar**
1. Both are periodic planning loops that read XR input and produce joint-space targets.
- Yours: [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:266)
- Official Python core: [base_teleop_controller.py](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python/xrobotoolkit_teleop/common/base_teleop_controller.py:180)

2. Both use grip/deadman threshold to activate control and anchor-relative pose deltas.
- Yours (state + mapping): [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:306), [teleop_mapper.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_mapper.cpp:36)
- Official Python: [base_teleop_controller.py](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python/xrobotoolkit_teleop/common/base_teleop_controller.py:189)

3. Both solve IK each cycle and then command the robot in a separate loop.
- Yours IK solve: [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:132)
- Official Python IK solve: [base_teleop_controller.py](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python/xrobotoolkit_teleop/common/base_teleop_controller.py:225)

**Where it differs (important)**
1. Your planner is explicitly integrated with a teleop state machine (`DISCONNECTED/IDLE/ARMED/ACTIVE/FAULT`), the official sample is mostly trigger-driven without this state gating.
- [teleop_state_machine.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/teleop_state_machine.cpp:9)

2. You enforce packet-age health + workspace clamp + jump reject in planner; official sample does not have equivalent planner safety checks.
- [safety.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/safety.cpp:28)

3. Your IK is custom Jacobian DLS + nullspace + manipulability damping; official sample uses Placo task-based IK.
- [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:132)
- [base_teleop_controller.py](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python/xrobotoolkit_teleop/common/base_teleop_controller.py:122)

4. Your planner publishes into lock-free latest-value buffers consumed by a 1 kHz libfranka callback; official sample runs soft realtime threads (`sleep`) and directly commands UR RTDE.
- Your buffers: [common_types.h](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/common_types.h:188)
- Official thread pacing: [base_hardware_teleop_controller.py](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python/xrobotoolkit_teleop/common/base_hardware_teleop_controller.py:125)

5. Your planner now includes explicit joint target smoothing before execution.
- [franka_controller.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/cpp/teleop_bridge/franka_controller.cpp:206)

6. The “official” C++ sample in `third_party` is also not architecturally equivalent to yours; it directly runs UR `servoL` threads at ~17 ms, no separate state-machine planner + 1 kHz servo callback.
- [main.cpp](/home/radu/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Cpp/main.cpp:493)

**Verdict**
Your planner thread is **conceptually similar** to official teleop planning logic (activate -> map delta -> IK -> command), but **implementation-wise it is more structured and safety-oriented**, and tightly adapted to Franka’s 1 kHz control model.
