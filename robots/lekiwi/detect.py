# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a LeKiwi site YAML by probing the serial bus and the cameras.

Run on the robot's computer, with the ``lekiwi`` extra installed:

    python -m robots.lekiwi.detect --output /path/to/robot.yaml

Nothing moves: the wheel bus is only pinged (lerobot's
``FeetechMotorsBus.scan_port`` broadcasts a ping at every baud rate) and each
camera is opened once to confirm it produces a frame. Geometry and drive
limits are policy, not discovery, so they are written as conservative
defaults and named in the report as fields to review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

#: Wheel motor IDs (left, back, right). Must match ``WHEEL_IDS`` in
#: rlinf/robotics/parts/mobility/lekiwi.py at the installed RLinf revision.
EXPECTED_WHEEL_IDS = (7, 8, 9)

#: Camera device index range probed when no stable by-id names exist.
_PROBE_INDICES = range(4)

_PLACEHOLDER_PORT = "/dev/ttyACM0"
_PLACEHOLDER_SERIAL = "<front-camera-serial>"


def list_serial_ports(dev_root: Path = Path("/dev")) -> list[str]:
    """Candidate wheel-bus ports, most stable spelling first.

    ``/dev/serial/by-id`` names survive re-enumeration, so they are preferred
    over the plain ``/dev/ttyACM*`` / ``/dev/ttyUSB*`` nodes they point at.
    """
    candidates: list[str] = []
    by_id = dev_root / "serial" / "by-id"
    if by_id.is_dir():
        candidates.extend(str(entry) for entry in sorted(by_id.iterdir()))
    for pattern in ("ttyACM*", "ttyUSB*", "tty.usb*"):
        candidates.extend(str(entry) for entry in sorted(dev_root.glob(pattern)))
    unique: dict[str, str] = {}
    for path in candidates:
        unique.setdefault(_realpath(path), path)
    return list(unique.values())


def _realpath(path: str) -> str:
    import os

    return os.path.realpath(path)


def scan_wheel_bus(port: str) -> dict[int, list[int]]:
    """Ping every baud rate on ``port``; returns ``{baudrate: [motor ids]}``."""
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus.scan_port(port)


def pick_wheel_port(
    ports: list[str],
) -> tuple[str | None, dict[str, dict[int, list[int]]]]:
    """Return the port whose bus answers for every wheel, plus all scan reports.

    A port where only some wheel ids answer is reported but not chosen: the
    env server's bus handshake pings every wheel, so a partial port would fail
    at startup anyway. A ``None`` port means no candidate is the wheel bus.
    """
    reports: dict[str, dict[int, list[int]]] = {}
    for port in ports:
        try:
            found = scan_wheel_bus(port)
        except Exception:  # noqa: BLE001 - a busy or absent port is a result
            found = {}
        if found:
            reports[port] = found
    expected = set(EXPECTED_WHEEL_IDS)

    def wheels_found(port: str) -> int:
        ids = {motor_id for ids in reports[port].values() for motor_id in ids}
        return len(expected & ids)

    for port in reports:
        if wheels_found(port) == len(expected):
            return port, reports
    return None, reports


def list_cameras(dev_root: Path = Path("/dev")) -> list[dict[str, str]]:
    """One entry per UVC camera, keyed by its stable by-id capture node.

    Each camera exposes several ``/dev/v4l2/by-id`` nodes; only the
    ``-index0`` node captures frames, so that is the one to configure.
    """
    by_id = dev_root / "v4l2" / "by-id"
    if not by_id.is_dir():
        return []
    cameras = []
    for entry in sorted(by_id.iterdir()):
        if not entry.name.endswith("-index0"):
            continue
        label = entry.name.removeprefix("usb-").removesuffix("-index0")
        cameras.append({"path": str(entry), "label": label})
    return cameras


def probe_camera(path: str) -> dict[str, Any]:
    """Open the camera once and report whether it produces a frame."""
    import cv2

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return {"path": path, "ok": False}
        ok, frame = capture.read()
        report: dict[str, Any] = {
            "path": path,
            "ok": bool(ok and frame is not None),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return report
    finally:
        capture.release()


def probe_camera_indices() -> list[dict[str, str]]:
    """Fall back to cv2 device indices when no by-id names exist (macOS, some UVC)."""
    found = []
    for index in _PROBE_INDICES:
        if probe_camera(str(index))["ok"]:
            found.append({"path": str(index), "label": f"camera {index}"})
    return found


def build_site_config(
    port: str | None, cameras: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the site YAML structure from the detection results."""
    devices: dict[str, dict[str, Any]] = {}
    for index, camera in enumerate(cameras):
        name = "front" if index == 0 else f"extra_{index}"
        devices[name] = {
            "serial": camera["path"],
            "type": "webcam",
            "main": index == 0,
        }
    if not devices:
        # Keep the file loadable: the placeholder serial fails at env start
        # with a clear "Failed to open webcam" instead of a schema error.
        devices["front"] = {
            "serial": _PLACEHOLDER_SERIAL,
            "type": "webcam",
            "main": True,
        }
    return {
        "robot": {
            "port": port or _PLACEHOLDER_PORT,
            "wheel_radius": 0.05,
            "base_radius": 0.125,
            "max_raw": 3000,
        },
        "cameras": {"devices": devices},
        "drive": {
            "step_frequency": 10.0,
            "max_linear_velocity": 0.3,
            "max_angular_velocity": 60.0,
            "frame_size": [128, 128],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect the LeKiwi wheel bus and cameras; write a site YAML."
    )
    parser.add_argument(
        "--output", default="lekiwi-site.yaml", help="path of the YAML to write"
    )
    parser.add_argument(
        "--skip-motors",
        action="store_true",
        help="do not ping the serial bus; emit a placeholder port",
    )
    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="do not open the cameras; emit a placeholder serial",
    )
    args = parser.parse_args(argv)

    ports: list[str] = []
    wheel_port: str | None = None
    bus_reports: dict[str, dict[int, list[int]]] = {}
    if not args.skip_motors:
        ports = list_serial_ports()
        wheel_port, bus_reports = pick_wheel_port(ports)

    cameras: list[dict[str, Any]] = []
    probed: list[dict[str, Any]] = []
    if not args.skip_cameras:
        cameras = list_cameras()
        if not cameras and sys.platform != "linux":
            cameras = probe_camera_indices()
        for camera in cameras:
            report = probe_camera(camera["path"])
            probed.append(report)
        cameras = [
            camera
            for camera in cameras
            if any(
                report["ok"] and report["path"] == camera["path"] for report in probed
            )
        ]

    config = build_site_config(wheel_port, cameras)

    print("== LeKiwi site detection ==")
    if not args.skip_motors:
        print(f"serial ports scanned: {ports or 'none found'}")
        for port, found in bus_reports.items():
            ids = sorted({motor_id for ids in found.values() for motor_id in ids})
            print(f"  {port}: motor ids {ids}")
        if wheel_port is None:
            print(
                "no port answered for wheel ids "
                f"{list(EXPECTED_WHEEL_IDS)}; check the USB cable and power, "
                "or pass --skip-motors to emit a placeholder"
            )
        else:
            print(f"wheel bus: {wheel_port}")
    if not args.skip_cameras:
        if probed:
            for report in probed:
                size = f"{report.get('width')}x{report.get('height')}"
                status = "ok" if report["ok"] else "NO FRAME"
                print(f"camera {report['path']}: {status} at {size}")
        else:
            print(
                "no cameras found under /dev/v4l2/by-id; run "
                "`v4l2-ctl --list-devices` and fill cameras.devices.front.serial"
            )

    output = Path(args.output)
    _write_yaml(output, config)
    print(f"wrote {output}")

    print("review before the first run:")
    print("  - robot.wheel_radius / robot.base_radius: measure with a ruler (m)")
    print("  - drive.max_linear_velocity / max_angular_velocity: policy; start low")
    print("  - drive.frame_size: camera output size; raise from 128x128 if needed")
    print("  - drive.step_frequency: control rate; 10 Hz suits the STS3215 bus")
    return 0


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
