# Franka VLA Teleoperation & Deployment

> XR teleoperation, data collection, and closed-loop Vision-Language-Action (VLA) policy deployment on a 7-DOF Franka Emika Panda, for autonomous electrical connector (MSD plug) insertion.

<img src="media/franka_smolvla_640_30.gif" alt="SmolVLA policy controlling the Franka Panda for MSD plug insertion" width="640">

## At a Glance

| Area | Details |
|------|---------|
| Robot control | Franka Emika Panda, libfranka 0.9.x, 1 kHz C++17 bridge |
| Policy deployment | Python 3.12/3.13 runner for LeRobot policies: SmolVLA, ACT, π0 |
| Teleoperation | Meta Quest 3 controller streaming through XRoboToolkit |
| Perception | Intel RealSense D405 and/or Stereolabs ZED-M camera feeds |

---

## Table of Contents

- [Setup](#setup)
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Repo Structure](#repo-structure)
- [Pipeline](#pipeline)
  - [1. Teleoperation & Data Collection](#1-teleoperation--data-collection)
  - [2. Data Cleaning & Conversion](#2-data-cleaning--conversion)
  - [3. Policy Deployment](#3-policy-deployment)
- [Implementation Details](#implementation-details)

---

## Setup

This repo expects a Franka Emika Panda setup with `libfranka` 0.9.x, a real-time-capable Linux kernel, XRoboToolkit PC Service, and the relevant camera SDKs installed on the robot host. For the lower-level Franka setup, see [`franka-sanity-checks/INSTALL.md`](franka-sanity-checks/INSTALL.md). Run the commands below from the repo root.

Clone the external dependencies:

```bash
git submodule update --init --recursive
```

Build the C++ bridge:

```bash
cmake -S franka_xr_teleop -B franka_xr_teleop/build -DCMAKE_BUILD_TYPE=Release \
  -DXROBOTICS_SERVICE_ROOT=/opt/apps/roboticsservice
cmake --build franka_xr_teleop/build -j"$(nproc)"
```

If the XRoboToolkit SDK is not in the default service tree, also pass `-DXROBOTICS_SDK_ROOT=/path/to/SDK`.

Sync the LeRobot policy environment:

```bash
uv python install 3.13
uv sync --project lerobot --python 3.13 \
  --extra smolvla \
  --extra intelrealsense
```

Use Python 3.12 instead if the installed ZED SDK does not provide a `pyzed` wheel for Python 3.13. ZED/ZED-M deployments also need the ZED SDK Python bindings installed into the LeRobot environment; see [`franka_xr_teleop/LEROBOT_VENV_SETUP.md`](franka_xr_teleop/LEROBOT_VENV_SETUP.md) for that camera-specific step.

From the repo root, verify the two main entry points:

```bash
./franka_xr_teleop/build/cpp/teleop_bridge/franka_xr_teleop_bridge --dry-run
```

```bash
uv run --project lerobot --python 3.13 \
  --extra smolvla \
  --extra intelrealsense \
  python franka_xr_teleop/tools/run_vla_policy.py --help
```

---

## Overview

This project deploys learned Vision-Language-Action (VLA) policies on a real robot arm to perform a contact-rich industrial task: grasping a Manual Service Disconnect (MSD) plug and inserting it into its socket. MSD plugs are used to isolate high-voltage systems during maintenance, and the task demands sub-millimetre alignment that is difficult for human operators wearing thick insulating gloves, which makes it a strong candidate for robotic automation.

The system spans the full pipeline: collecting human demonstrations through XR teleoperation, processing them into a training dataset, fine-tuning VLA policies, and deploying those policies closed-loop on the hardware. Three policy architectures are supported: SmolVLA, ACT, and π0, all from the [LeRobot](https://github.com/huggingface/lerobot) framework.

## System Architecture

The deployment splits across two processes to separate slow, GPU-bound model inference from the hard real-time control loop:

- A **Python policy runner** consumes robot observations, runs model inference, and streams joint targets.
- A **C++ bridge** (`franka_xr_teleop_bridge`) receives those targets and drives the arm via libfranka at 1 kHz, meeting the hard real-time constraints the controller requires.
- The two communicate over **UDP**: action packets out at 30 Hz, observations published back at ~50 Hz.

This separation is what lets a 450M–3.3B parameter model drive a real-time arm without stalling the control loop.

## Repo Structure

| Path | Description |
|------|-------------|
| `franka_xr_teleop/cpp/teleop_bridge/` | Real-time C++ bridge: XR/policy command sources, planner, safety validation, libfranka control loop |
| `franka_xr_teleop/tools/run_vla_policy.py` | The deployment runner: loads a LeRobot policy and streams joint targets to the bridge over UDP |
| `franka_xr_teleop/tools/record_data_collection_session.py` | Launches synchronised robot + camera recorders for one collection session |
| `franka_xr_teleop/tools/record_robot_observations.py` | Records UDP robot observations and episode markers to JSONL |
| `franka_xr_teleop/tools/record_realsense_camera.py`, `franka_xr_teleop/tools/record_zed_camera.py` | Timestamped camera recorders for dataset sync |
| `franka_xr_teleop/tools/align_robot_camera_jsonl.py` | Nearest-neighbour timestamp alignment of robot state to camera frames |
| `franka_xr_teleop/tools/test_deploy_paths.py` | Hardware-free harness exercising both inference paths on CPU |
| `franka_xr_teleop/tools/plot_*.py`, `franka_xr_teleop/tools/live_teleop_debug.py` | Diagnostics: trajectory plots, live UDP observation monitor |

## Pipeline

### 1. Teleoperation & Data Collection

A Meta Quest 3 controller drives the arm through inverse kinematics over UDP, with a deadman's switch gating motion and controller buttons marking episode start/end. Two synchronised camera feeds (wrist and third-person) and per-episode robot state are recorded to JSONL.

### 2. Data Cleaning & Conversion

Raw episodes are trimmed of dead time, filtered for meaningful motion, and timestamp-matched (camera frames to the nearest robot sample). The cleaned data is converted into the LeRobotDataset format (parquet state/action + re-encoded MP4 video) ready for fine-tuning.

### 3. Policy Deployment

The policy runner loads a fine-tuned checkpoint and streams absolute joint targets to the bridge. For flow-matching policies (SmolVLA, π0) it uses **Real-Time Chunking (RTC)**: a background thread runs inference asynchronously and blends each new action chunk into the previous one, holding a steady 30 Hz command stream regardless of inference latency. Output smoothing and joint-limit safety clamping are applied before each command is sent.

## Implementation Details

- **Cross-language real-time split.** GPU inference in Python, 1 kHz control in C++, decoupled over UDP so neither blocks the other.
- **Asynchronous RTC deployment.** The LeRobot inference APIs don't talk to the Franka bridge directly; the runner was built from scratch to produce action chunks asynchronously and stream them without gaps.
- **Action filtering for contact-rich control.** Light filtering (EMA) preserved the small corrective motions needed to seat the plug; heavier low-pass filtering removed them and insertion failed entirely.
- **Safety and operator control.** Joint-limit clamping, hold-on-dropped-packet behaviour, and live operator controls (pause, re-home, resume) for safe hardware operation.
- **Hardware-free testing.** `test_deploy_paths.py` exercises both inference paths on CPU with dummy observations, enabling iteration without the hardware connected.
