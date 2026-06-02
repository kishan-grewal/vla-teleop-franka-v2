# LeRobot Venv Setup

This note covers the full local setup flow for the `lerobot` environment used by
`franka_xr_teleop/tools/run_vla_policy.py`, including installing the
RealSense Python bindings into the same venv.

## Recommended Setup

Use Python `3.13` for this project right now if the installed ZED SDK provides
a matching `pyzed` wheel for it. If the ZED installer script cannot find a
`pyzed` package for Python `3.13`, use Python `3.12` instead.

Why:

- The local `lerobot` checkout requires Python `>=3.12`.
- Our policy runner currently blocks Python `3.14` because `draccus` config
parsing is failing in this workspace under `3.14`.
- Python `3.13` is a good default for this repo, but `pyzed` wheel availability
depends on the installed ZED SDK.

## Fresh Setup

Run this from the repo root:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
uv python install 3.13
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
uv pip install -e ".[smolvla,intelrealsense]"
```

What this does:

- creates a Python `3.13` virtual environment in `lerobot/.venv`
- seeds `pip` into the virtual environment
- installs the local `lerobot` checkout in editable mode
- installs the `smolvla` extra
- installs the `intelrealsense` extra, which pulls in `pyrealsense2`

Then install the ZED Python API into the same active venv:

```bash
source /home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/activate
cd /usr/local/zed
python3 get_python_api.py
```

If `/usr/local/zed/get_python_api.py` complains about permissions, copy it into
a writable directory and run it from there while the venv is active.

What this does:

- detects the active Python interpreter in the venv
- downloads the matching `pyzed` wheel for the installed ZED SDK
- installs `pyzed` into the current venv

## Rebuild An Existing Broken Venv

If you already created `lerobot/.venv` with Python `3.14`, rebuild it:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
rm -rf .venv
uv python install 3.13
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
uv pip install -e ".[smolvla,intelrealsense]"
cd /usr/local/zed
python3 get_python_api.py
```

## Add RealSense Bindings To An Existing Good Venv

If your `lerobot/.venv` is already Python `3.12` or `3.13`, you do not need to
recreate it. Just activate it and install the RealSense extra:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
source .venv/bin/activate
python --version
uv pip install -e ".[intelrealsense]"
```

If you prefer to install the package directly instead of the extra:

```bash
uv pip install pyrealsense2
```

## Add ZED Bindings To An Existing Good Venv

Make sure the ZED SDK is already installed on the machine. Then, with the
LeRobot venv activated:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
source .venv/bin/activate
cd /usr/local/zed
python3 get_python_api.py
```

If the script succeeds, `pyzed` will be installed into the active venv.

If the script says no matching wheel is available for your interpreter, rebuild
the venv with Python `3.12` and rerun it:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
rm -rf .venv
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -e ".[smolvla,intelrealsense]"
cd /usr/local/zed
python3 get_python_api.py
```

## Verify The Environment

With the venv activated:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy
./lerobot/.venv/bin/python --version
./lerobot/.venv/bin/python -c "import pyrealsense2 as rs; print('pyrealsense2 import OK')"
./lerobot/.venv/bin/python -c "import pyzed.sl as sl; print('pyzed import OK')"
./lerobot/.venv/bin/python -c "from lerobot.policies.smolvla import SmolVLAPolicy; print('SmolVLA import OK')"
```

## Running The Policy Runner

After the venv is set up:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy
./lerobot/.venv/bin/python franka_xr_teleop/tools/run_vla_policy.py --help
```

List both RealSense and ZED serials directly from the main runner:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy
source ./lerobot/.venv/bin/activate
python franka_xr_teleop/tools/run_vla_policy.py --list-cameras
```

Example output:

```text
realsense_device_count=1
realsense[0]: name=Intel RealSense D405 serial=128422271175 firmware=... usb=...
zed_device_count=1
zed[0]: serial=12345678 model=ZED 2i state=...
```

Example launch:

```bash
source ./lerobot/.venv/bin/activate
python franka_xr_teleop/tools/run_vla_policy.py \
  --policy-path ./model/pretrained_model \
  --obs-port 28081 \
  --bridge-ip 127.0.0.1 \
  --action-port 28082 \
  --top-camera-backend zed-left \
  --third-person-camera-backend realsense \
  --realsense-serial <REALSENSE SERIAL> \
  --zed-serial <ZED SERIAL>  \
  --task "your task"
```

## Common Problems

### Python 3.14

Symptom:

```text
TypeError: typing.Dict[...] | None is not callable
```

Fix:

- recreate the venv with Python `3.13`

### `ImportError: No module named pyrealsense2`

Fix:

```bash
cd /home/radu/vla-teleop-franka-v2-model-deploy/lerobot
source .venv/bin/activate
uv pip install -e ".[intelrealsense]"
```

### `ImportError: No module named pyzed.sl`

Fix:

```bash
source /home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/activate
cd /usr/local/zed
python3 get_python_api.py
```

### `No module named pip`

Symptom:

```text
/path/to/python: No module named pip
```

Fix for an existing venv:

```bash
/home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/python -m ensurepip --upgrade
```

Fix when creating a new venv:

```bash
uv venv --python 3.13 --seed .venv
```

### `pyzed` Install Fails

Possible causes:

- the ZED SDK is not installed
- the active Python version does not have a matching wheel for your installed
ZED SDK
- the script was run outside the intended venv

Fixes:

- verify `/usr/local/zed/get_python_api.py` exists
- activate `lerobot/.venv` first
- if Python `3.13` is unsupported by the current ZED SDK, recreate the venv
with Python `3.12` and rerun the script

### `pyrealsense2` Install Fails

If `uv pip install pyrealsense2` or `uv pip install -e ".[intelrealsense]"` fails
because no wheel is available for your interpreter/platform, install the Intel
RealSense SDK and Python bindings using the official `librealsense` instructions,
then retry the import in the same venv.

## Useful One-Liners

Activate the environment:

```bash
source /home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/activate
```

Check Python version:

```bash
/home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/python --version
```

List RealSense devices:

```bash
/home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/python \
  /home/radu/vla-teleop-franka-v2-model-deploy/franka_xr_teleop/tools/record_realsense_camera.py \
  --list-devices
```

List both RealSense and ZED devices from the main runner:

```bash
/home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/python \
  /home/radu/vla-teleop-franka-v2-model-deploy/franka_xr_teleop/tools/run_vla_policy.py \
  --list-cameras
```

Install `pyzed` into the active venv:

```bash
source /home/radu/vla-teleop-franka-v2-model-deploy/lerobot/.venv/bin/activate
cd /usr/local/zed
python3 get_python_api.py
```
