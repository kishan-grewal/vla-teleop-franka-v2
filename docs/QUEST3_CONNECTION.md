# Quest 3 Connection for Teleoperation

This guide covers how to connect a Quest 3 headset to the PC teleoperation stack using ADB over USB.

## 1) Prerequisites

- Quest 3 Developer Mode is enabled.
- USB debugging is enabled/allowed on the headset.
- `adb` is installed on the PC.
- A USB data cable is connected between Quest 3 and PC.
- XRoboToolkit PC service is installed on the PC.

## 2) Authorize the headset with ADB

Run:

```bash
adb kill-server
adb start-server
adb devices
```

Then put on the headset and accept:

- `Allow USB debugging?`
- Check `Always allow from this computer`.

Expected `adb devices` state:

- Quest device appears as `device` (not `unauthorized`).

## 3) If `adb devices` shows no devices

Symptom:

- `adb devices` returns an empty list.
- `adb reverse ...` fails with `error: no devices/emulators found`.

Steps:

1. Reconnect with a known USB data path:

- Keep Quest powered on and unlocked.
- Replug USB directly to laptop (avoid hubs/docks during debugging).
- Try a different USB port and a known data-capable cable (not charge-only).

1. Re-check device detection:

```bash
adb kill-server
adb start-server
adb devices -l
```

1. Confirm Linux sees Quest on USB:

```bash
lsusb | grep -Ei 'meta|oculus|android|quest'
```

Interpretation:

- If `lsusb` shows nothing relevant, this is a cable/port/USB-path issue.
- If `lsusb` shows Quest but `adb devices` is still empty, continue with authorization reset in the next section.

### 3.1) If `lsusb` sees Quest but `adb devices -l` is empty (common after replug)

Symptom:

- `lsusb` shows `ID 2833:0183 Oculus Quest 3`.
- `adb devices -l` is empty.
- You checked `/dev/bus/usb/<bus>/<dev>` and it did not exist.

Note:

- The USB device number can change after reconnect. Re-check current `BUS/DEV` every time.

Get current device node dynamically:

```bash
lsusb -d 2833:0183
BUS=$(lsusb -d 2833:0183 | awk '{print $2}')
DEV=$(lsusb -d 2833:0183 | awk '{print $4}' | tr -d :)
echo "/dev/bus/usb/$BUS/$DEV"
ls -l /dev/bus/usb/$BUS/$DEV
```

Reset ADB + host keys and re-authorize:

```bash
pkill -f "adb.*server" || true
rm -f ~/.android/adbkey ~/.android/adbkey.pub
adb kill-server
adb start-server
adb devices -l
```

On Quest:

- Wear headset and keep it unlocked on Home.
- `Settings -> System -> Developer`.
- `Revoke USB debugging authorizations`.
- Toggle `USB debugging` Off then On.
- Replug USB and accept `Allow USB debugging`.

If still empty, test permission path:

```bash
sudo adb kill-server
sudo adb start-server
sudo adb devices -l
```

If `sudo adb` works but normal `adb` does not, fix udev permissions:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/51-android.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG plugdev $USER
```

Then log out/in, unplug/replug Quest, and retry:

```bash
adb kill-server
adb start-server
adb devices -l
```

## 4) If `adb devices` shows `unauthorized`

1. On Quest 3:

- Go to `Settings -> System -> Developer`.
- Select `Revoke USB debugging authorizations`.

1. Reconnect USB and restart ADB:

```bash
adb kill-server
adb start-server
adb devices
```

1. If still unauthorized, regenerate host ADB keys:

```bash
rm -f ~/.android/adbkey ~/.android/adbkey.pub
adb kill-server
adb start-server
adb devices
```

Accept the debugging prompt again inside the headset.

## 5) Enable USB tunnel to PC service

Run:

```bash
adb reverse --remove-all
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

Expected:

- `adb reverse --list` includes `tcp:63901 tcp:63901`.

In the Quest app, connect to:

```text
127.0.0.1
```

## 6) Start teleoperation stack

Start XRoboToolkit PC service (example):

```bash
/opt/apps/roboticsservice/run3D.sh
```

Then run the bridge from this repo (example real robot):

```bash
cd /home/radu/vla-teleop-franka-v2/franka_xr_teleop
./build/cpp/teleop_bridge/franka_xr_teleop_bridge --robot-ip 192.168.2.200 --obs-port 28081
```

## 7) Quick validation checklist

- `adb devices` shows Quest as `device`.
- `adb reverse --list` shows `tcp:63901 tcp:63901`.
- Quest app connects to `127.0.0.1`.
- Bridge logs show incoming XR/controller data.

