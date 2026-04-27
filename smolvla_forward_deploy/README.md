# SmolVLA Forward Deploy for Franka

This folder is a deployment scaffold for running forward inference of a trained SmolVLA checkpoint on Franka observations.

## What it does

- Subscribes to Franka observation packets from `franka_xr_teleop_bridge` (`--obs-port`).
- Builds SmolVLA inputs (`observation.state` and one image stream).
- Runs policy forward pass.
- Publishes predicted Franka actions to UDP.
- Optionally logs every action packet to JSONL.

## Built-in Franka Failsafes

The runtime now enforces safety guards by default for a 7DOF Franka Panda:

- Hard Panda joint bounds with extra safety margin.
- Per-step joint delta clamp (`max_joint_step_rad`).
- Velocity-based clamp (`max_joint_speed_rad_s`).
- Gripper width clamp (`gripper_width_min_m`, `gripper_width_max_m`).
- Observation freshness gate (`max_observation_age_ms`).
- UDP timeout watchdog (`socket_timeout_ms`) with hold behavior.
- Fail-closed behavior on missing observation timestamps.
- Inference exception fallback to hold action.
- Non-monotonic observation sequence rejection.
- Status/fault gate using teleop fault flags (`packet_timeout`, `jump_rejected`, `workspace_clamped`, `robot_not_ready`, `control_exception`, `ik_rejected`).
- Joint target smoothing aligned with teleop controller logic (velocity, acceleration, and alpha blending).
- Arming gate file (`ARMED`) required before active commands.
- File-based emergency stop (`ESTOP`) forcing hold output.

All hold modes publish a safe command that keeps current joints and gripper width.

## Files

- `run_forward_deploy.py`: main deployment runtime.
- `deploy_config.example.json`: editable runtime config template.
- `action_command.schema.json`: action payload schema emitted by this runtime.

## Important integration note

The existing C++ teleop bridge currently publishes observations but does not consume model actions directly.
This deployment folder therefore provides:

1. Action UDP output (`action_udp_host`, `action_udp_port`) for your next control bridge step.
2. JSONL logging of action packets for validation and replay.

## Usage

1. Copy and edit config:

   - Duplicate `deploy_config.example.json` to `deploy_config.json`.
   - Set `model_path` to your uploaded fine-tuned SmolVLA checkpoint.
   - Set `lerobot_root` to your local lerobot checkout path if needed.
   - If you have a camera pipeline ready, set `camera.mode`.

2. Arm/E-stop procedure (recommended):

   - Keep `safety.require_armed_file=true`.
   - Do not create `ARMED` until robot is clear and supervised.
   - Create `ESTOP` at any time to force hold output.
   - Remove `ESTOP` only after verifying the scene is safe.

3. Start Franka bridge observation publisher (example):

   - `./build/cpp/teleop_bridge/franka_xr_teleop_bridge --robot-ip <ROBOT_IP> --obs-port 28081`

4. Run forward deploy:

   - `python smolvla_forward_deploy/run_forward_deploy.py --config smolvla_forward_deploy/deploy_config.json`

## Camera input

Current camera modes:

- `dummy`: synthetic image tensor (good for pipeline bring-up).
- `webcam`: pulls one frame per step from OpenCV webcam index.

For production deployment, replace camera input with your real Franka camera stream and match keys expected by your trained checkpoint.

## Action format

The runtime emits packets matching `action_command.schema.json` with:

- `joint_targets_q1_q7`
- `gripper_width`
- `raw_action`
- `safety` metadata (`mode`, `reason`, `clipped_indices`, `observation_age_ms`)

## Next step after weight upload

When you upload weights, update only `model_path` in your config and run the same command.

For first real-robot tests, start with conservative values (already set):

- `max_joint_step_rad = 0.008`
- `max_joint_speed_rad_s = 0.35`
- `max_joint_acceleration_rad_s2 = 1.5`
- `target_smoothing_alpha = 0.25`
- `joint_position_margin_rad = 0.10`

Teleop-compatibility toggles in `safety`:

- `blocked_fault_flags`
- `enforce_monotonic_sequence`
- `require_target_fresh`
- `require_teleop_active`
- `allowed_control_modes`
