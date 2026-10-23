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

"""Offline tests for hold-to-drive teleoperation and its recordings."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rpent.dashboard.interaction import InteractionUnavailableError
from rpent.dashboard.teleop import ACTION_LOG_NAME, TeleopController

CHUNK_S = 0.05


class FakeEnv:
    """Records each drive command and returns one synthetic camera frame."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, float, float], float]] = []
        self.stop_calls = 0
        self.observations = 0
        self.drove = threading.Event()
        self._lock = threading.Lock()

    def drive(self, velocity: Any, *, duration_s: float) -> dict[str, Any]:
        commanded = np.asarray(velocity, dtype=np.float32)
        with self._lock:
            self.calls.append((tuple(float(v) for v in commanded), duration_s))
        self.drove.set()
        # The real env spends the chunk stepping, which paces the worker.
        time.sleep(duration_s)
        return {
            "ok": True,
            "commanded_velocity": commanded.tolist(),
            "steps": 2,
            "duration_s": duration_s,
        }

    def stop(self) -> dict[str, Any]:
        self.stop_calls += 1
        return {"ok": True}

    def get_observation(self) -> dict[str, Any]:
        self.observations += 1
        frame = np.full((4, 5, 3), self.observations % 255, dtype=np.uint8)
        return {"states": np.zeros(3, dtype=np.float32), "main_images": frame}

    def get_env_meta(self) -> dict[str, Any]:
        return {"velocity_limits": [0.3, 0.3, 60.0]}

    def get_camera_meta(self) -> dict[str, Any]:
        return {"observation_camera_map": {"main": "front"}}


class FakeDashboard:
    """Stand-in for DashboardState's teleop sinks."""

    def __init__(self) -> None:
        self.frames: list[dict[str, bytes]] = []
        self.actions: list[dict[str, Any]] = []

    def publish_live_frames(self, frames: dict[str, bytes]) -> int:
        self.frames.append(dict(frames))
        return len(self.frames)

    def record_teleop_action(self, item: dict[str, Any]) -> None:
        self.actions.append(item)


class FakeEnvState:
    """Capture run-level artifacts instead of writing mp4 files."""

    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []

    def save(self, name: str, value: Any, *, step: Any = None, fps: Any = None):
        self.saves.append(
            {"name": name, "frames": list(value), "step": step, "fps": fps}
        )
        return name


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met within the timeout")


def _wait_until_steps_settle(
    controller: TeleopController, quiet_s: float = 0.25
) -> int:
    """Block until the drive worker has stopped adding steps."""
    last = controller.snapshot()["steps"]
    since = time.monotonic()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        time.sleep(0.02)
        steps = controller.snapshot()["steps"]
        if steps != last:
            last, since = steps, time.monotonic()
        elif time.monotonic() - since >= quiet_s:
            return steps
    raise AssertionError("the drive worker never settled")


@pytest.fixture
def rig(tmp_path: Path):
    env, dashboard, env_state = FakeEnv(), FakeDashboard(), FakeEnvState()
    controllers: list[TeleopController] = []

    def build(**kwargs: Any) -> TeleopController:
        controller = TeleopController(
            env=env,
            dashboard=dashboard,
            output_dir=tmp_path,
            env_state=env_state,
            chunk_s=CHUNK_S,
            **kwargs,
        )
        controllers.append(controller)
        return controller

    yield build, env, dashboard, env_state
    for controller in controllers:
        controller.close()


def test_hold_drives_chunks_and_records_video_actions_and_frames(rig, tmp_path):
    build, env, dashboard, env_state = rig
    controller = build()
    controller.hold([0.1, -0.05, 10.0], ttl_s=0.12)
    assert env.drove.wait(timeout=3)
    _wait_until(lambda: len(dashboard.actions) >= 2)

    summary = controller.close()

    assert env.calls, "holding a velocity must drive the env"
    assert all(
        velocity == pytest.approx((0.1, -0.05, 10.0)) for velocity, _ in env.calls
    )
    assert all(0 < duration <= CHUNK_S for _, duration in env.calls)
    assert env.stop_calls == 1
    assert dashboard.frames, "live frames are shown even when not recorded"
    assert set(dashboard.frames[0]) == {"front"}
    assert summary["steps"] == len(env.calls)
    assert [clip["name"] for clip in summary["clips"]] == ["teleop_000.mp4"]
    saved = env_state.saves[0]
    assert saved["step"] is None
    assert saved["fps"] == round(1 / CHUNK_S)
    assert len(saved["frames"]) == len(env.calls)
    lines = (tmp_path / ACTION_LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(env.calls)
    assert json.loads(lines[0])["velocity"] == [0.1, -0.05, 10.0]


def test_an_expired_command_stops_the_worker_from_driving(rig):
    build, env, _, _ = rig
    controller = build()
    controller.hold([0.2, 0.0, 0.0], ttl_s=0.06)
    assert env.drove.wait(timeout=3)
    settled = _wait_until_steps_settle(controller)
    time.sleep(0.3)
    snapshot = controller.snapshot()
    assert snapshot["steps"] == settled, "an expired command must not keep driving"
    assert snapshot["velocity"] == [0.0, 0.0, 0.0]

    # A fresh hold takes over without the worker having to be restarted.
    controller.hold([0.2, 0.0, 0.0], ttl_s=2.0)
    _wait_until(lambda: controller.snapshot()["steps"] > settled)


def test_recording_off_keeps_driving_but_writes_nothing(rig, tmp_path):
    build, env, dashboard, env_state = rig
    controller = build()
    controller.set_recording(video=False, actions=False)
    controller.hold([0.1, 0.0, 0.0], ttl_s=0.08)
    assert env.drove.wait(timeout=3)
    _wait_until(lambda: len(dashboard.frames) >= 2)

    snapshot = controller.set_recording(video=False, actions=False)
    controller.close()

    assert snapshot["recording"] == {"video": False, "actions": False}
    assert dashboard.frames, "the operator still needs the live camera"
    assert env_state.saves == []
    assert dashboard.actions == []
    assert (tmp_path / ACTION_LOG_NAME).read_text(encoding="utf-8") == ""


def test_turning_video_off_flushes_the_frames_collected_so_far(rig):
    build, env, dashboard, env_state = rig
    controller = build()
    controller.hold([0.1, 0.0, 0.0], ttl_s=2.0)
    assert env.drove.wait(timeout=3)
    _wait_until(lambda: len(dashboard.frames) >= 3)

    snapshot = controller.set_recording(video=False)

    assert snapshot["recording"]["video"] is False
    assert [clip["name"] for clip in snapshot["clips"]] == ["teleop_000.mp4"]
    assert len(env_state.saves) == 1


def test_hold_rejects_a_velocity_that_is_not_three_finite_numbers(rig):
    build, _, _, _ = rig
    controller = build()

    for bad in ([[0.1, 0.0, 0.0]], [0.1, 0.0], [float("nan"), 0.0, 0.0]):
        with pytest.raises(ValueError, match="3 finite numbers"):
            controller.hold(bad)
    with pytest.raises(ValueError, match="ttl_s"):
        controller.hold([0.1, 0.0, 0.0], ttl_s=0)


def test_close_stops_the_base_and_rejects_further_commands(rig):
    build, env, _, _ = rig
    controller = build()
    controller.hold([0.1, 0.0, 0.0], ttl_s=2.0)
    assert env.drove.wait(timeout=3)

    controller.close()

    assert env.stop_calls == 1
    assert controller.snapshot()["active"] is False
    with pytest.raises(InteractionUnavailableError):
        controller.hold([0.1, 0.0, 0.0])
