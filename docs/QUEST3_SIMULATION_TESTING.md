# QUEST3 Simulation Testing

## Goal
Validate our Quest 3 port and the official XRoboToolkit Quest workflow in simulation (non-ROS) before any real robot test.

## Daily Restart Quickstart (Copy/Paste)

Use this each time PC/Quest has been restarted.

### 1) USB tunnel setup (Quest connected by USB)

```bash
adb reverse --remove-all
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

In Quest app, connect to `127.0.0.1`.

### 2) Start PC service

```bash
pkill -f RoboticsServiceProcess
/opt/apps/roboticsservice/run3D.sh
```

### 3) Start simulation backend

```bash
cd /home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python
conda activate xr
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```

### 4) Quest app toggles

- `Controller` ON
- `Send` ON (or `Switch w/ A Button` ON and press A)
- Hold grips to move arms, use triggers for grippers

## 1) Install XRoboToolkit PC Service on Ubuntu 22.04 (amd64)

Check architecture:

```bash
uname -m
dpkg --print-architecture
```

Expected: `x86_64` and `amd64`.

Install PC service:

```bash
cd ~/Downloads
wget -O XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb \
  https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo apt update
sudo apt install -y ./XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

If apt reports dependency issues:

```bash
sudo apt -f install -y
sudo apt install -y ./XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Service run command:

```bash
/opt/apps/roboticsservice/runService.sh
```

## 2) Ensure XR submodules are present in this repo

```bash
cd /home/rechim/vla-teleop-franka-v2
git submodule sync --recursive
git submodule update --init --recursive
```

Expected third-party repos:
- `XRoboToolkit-PC-Service`
- `XRoboToolkit-PC-Service-Pybind`
- `XRoboToolkit-Unity-Client`
- `XRoboToolkit-Unity-Client-Quest`
- `XRoboToolkit-Teleop-Sample-Python`
- `XRoboToolkit-Teleop-Sample-Cpp`

## 3) Prepare simulation Python environment

```bash
cd /home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python
bash setup_conda.sh --conda xr
conda activate xr
bash setup_conda.sh --install
```

Important fix for user-site package leakage:

```bash
conda activate xr
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate xr
```

## 4) Install Unity for Quest app build

Use Unity Hub and install exactly:
- Unity `2021.3.45f2`

Select modules:
- `Android Build Support`
- `Android SDK & NDK Tools`
- `OpenJDK`

No other modules are required for this flow.

## 5) Build Quest APK from official Quest client project

Project path:

```text
/home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Unity-Client-Quest
```

In Unity:
1. Open the project with Unity `2021.3.45f2`.
2. Open `File -> Build Settings`.
3. Select `Android` and click `Switch Platform` if needed.
4. Click `Build`.
5. In the save dialog, set file name to `XRoboToolkit-Quest.apk` and save in project root.
6. If prompted with `Unsupported Input Handling`, click `Yes` and continue.

Expected APK path:

```text
/home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Unity-Client-Quest/XRoboToolkit-Quest.apk
```

## 6) Install APK to Quest 3

Prerequisites:
- Quest Developer Mode enabled
- USB debugging authorized on headset

Install command:

```bash
adb devices
adb install -r /home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Unity-Client-Quest/XRoboToolkit-Quest.apk
```

If APK path differs:

```bash
find /home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Unity-Client-Quest -name "*.apk"
```

Launch app from Quest `Unknown Sources` as `XRoboToolkit-Quest`.

## 7) Run teleop simulation

Terminal A:

```bash
pkill -f RoboticsServiceProcess
/opt/apps/roboticsservice/run3D.sh
```

Terminal B:

```bash
cd /home/rechim/vla-teleop-franka-v2/franka_xr_teleop/third_party/XRoboToolkit-Teleop-Sample-Python
conda activate xr
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```

### Mode A: LAN connection (same local network)

Get PC IP for Quest connection:

```bash
hostname -I
```

Use the Wi-Fi IP (not Docker interfaces like `172.x.x.x`).

Quest app settings:
- Connect to PC service IP (example: `10.23.114.44`)
- `Controller` ON
- `Send` ON
- Optional: `Switch w/ A Button` ON
- Hold grips to teleoperate
- Use triggers for gripper control

### Mode B: USB tunnel fallback (works even when Wi-Fi blocks peer traffic)

If Quest shows `connect_error` on LAN, keep Quest connected via USB and run:

```bash
adb reverse --remove-all
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

Then in Quest app, connect to:

```text
127.0.0.1
```

This forwards Quest local TCP port `63901` over USB to the laptop service.

## 8) Expected success signals

- Python sim logs `XRoboToolkit SDK initialized.`
- Connection logs appear (`connect127.0.0.1:60061`, `server connect`)
- Pose tasks for left/right hands are created
- Viewer URL is printed (`http://127.0.0.1:7000/static/`)
- Quest app shows connected/working
- MuJoCo arms move with Quest controller input

Note: neutral self-collision warnings from the demo model are expected and non-blocking.

## 9) Troubleshooting

### A) Unity build fails with script compile errors and log shows:
`No usable version of libssl was found`

Fix on Ubuntu 22.04:

```bash
echo "deb http://security.ubuntu.com/ubuntu focal-security main" | sudo tee /etc/apt/sources.list.d/focal-security.list
sudo apt update
sudo apt install -y libssl1.1
sudo rm /etc/apt/sources.list.d/focal-security.list
sudo apt update
```

Then restart Unity Hub + Unity Editor and rebuild.

If still failing and shell environment injects conflicting libs, start Unity Hub with clean `LD_LIBRARY_PATH`:

```bash
env -u LD_LIBRARY_PATH unityhub
```

### B) Quest cannot connect

Check:
- PC and Quest are on the same Wi-Fi network
- VPN is off on PC
- `run3D.sh` (or `runService.sh`) is running
- Manually enter PC IP in app if auto-discovery fails

If you are on `eduroam` or enterprise/campus Wi-Fi:
- Client-to-client traffic is often blocked.
- Prefer private hotspot/router, or use USB tunnel mode (`adb reverse`) and connect Quest app to `127.0.0.1`.

Quick hotspot note for iPhone:
- Enable `Personal Hotspot`
- Enable `Maximize Compatibility`
- Connect both laptop and Quest to that hotspot

### C) Python sim imports wrong user packages

Re-apply:

```bash
conda activate xr
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate xr
```

### D) APK installed but not visible in Quest

- Open `Apps` on Quest
- Switch filter to `Unknown Sources`
- Launch `XRoboToolkit-Quest`

Verify package via adb:

```bash
adb shell pm list packages | grep xrobotoolkit
```

## 10) Optional bridge compatibility check (this repo)

After building `franka_xr_teleop`, run:

```bash
./build/cpp/teleop_bridge/franka_xr_teleop_bridge --dry-run
```

This validates XR input ingestion through our bridge path without commanding a real robot.
