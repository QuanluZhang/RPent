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

"""Hold-to-drive teleoperation for one planner-free Dashboard TaskRun.

The browser sends a body velocity and keeps refreshing it while a control is
held. A worker thread turns that target into short ``drive`` chunks, so a
lost browser stops the base once the command expires: the robot's env stops
the wheels at the end of every control period, and this worker stops stepping.
"""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np

from rpent.dashboard.interaction import InteractionUnavailableError
from rpent.utils.logging import get_logger

logger = get_logger("dashboard_teleop")

# One chunk bounds command latency. The wheels stop at the end of every env
# control period, so this is also the longest a released command keeps moving.
DEFAULT_CHUNK_S = 0.2
DEFAULT_COMMAND_TTL_S = 1.0
MAX_COMMAND_TTL_S = 2.0
ACTION_LOG_NAME = "teleop_actions.jsonl"


class TeleopEnv(Protocol):
    """Robot env operations that manual driving needs."""

    def drive(self, velocity: Any, *, duration_s: float) -> dict[str, Any]:
        """Hold one body velocity for a bounded duration."""
        ...

    def stop(self) -> dict[str, Any]:
        """Command zero wheel velocity."""
        ...

    def get_observation(self) -> dict[str, Any]:
        """Read the live observation without commanding motion."""
        ...

    def get_env_meta(self) -> dict[str, Any]:
        """Return the action width and velocity limits."""
        ...

    def get_camera_meta(self) -> dict[str, Any]:
        """Return the camera identities behind the emitted frames."""
        ...


@dataclass(frozen=True)
class TeleopRecording:
    """Which artifacts one manual run keeps."""

    video: bool = True
    actions: bool = True

    def as_dict(self) -> dict[str, bool]:
        """Return the JSON-safe representation exposed to the frontend."""
        return {"video": self.video, "actions": self.actions}


class TeleopController:
    """Own one drive worker and the artifacts a manual TaskRun records."""

    def __init__(
        self,
        *,
        env: TeleopEnv,
        dashboard: Any,
        output_dir: str | Path,
        env_state: Any = None,
        chunk_s: float = DEFAULT_CHUNK_S,
    ) -> None:
        self._env = env
        self._dashboard = dashboard
        self._output_dir = Path(output_dir)
        self._env_state = env_state
        self._chunk_s = max(0.05, float(chunk_s))
        self._condition = threading.Condition()
        self._velocity = (0.0, 0.0, 0.0)
        self._commanded = (0.0, 0.0, 0.0)
        self._ttl_s = 0.0
        self._command_at = time.monotonic()
        self._recording = TeleopRecording()
        self._closed = False
        self._error: str | None = None
        self._steps = 0
        self._frames: list[np.ndarray] = []
        self._clips: list[dict[str, Any]] = []
        self._action_file: Any = None
        self._limits = (0.3, 0.3, 60.0)
        self._frame_key = "front"
        self._started_at = time.monotonic()
        self._read_environment()
        self._open_action_log()
        self._thread = threading.Thread(
            target=self._run, name="rpent-teleop", daemon=True
        )
        self._thread.start()

    # -- commands ----------------------------------------------------------

    def hold(
        self, velocity: Any, *, ttl_s: float = DEFAULT_COMMAND_TTL_S
    ) -> dict[str, Any]:
        """Set the target velocity; the caller refreshes it before it expires."""
        values = self._coerce_velocity(velocity)
        ttl = float(ttl_s)
        if not np.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_s must be finite and positive")
        with self._condition:
            self._raise_if_closed_locked()
            self._velocity = values
            self._ttl_s = min(ttl, MAX_COMMAND_TTL_S) if any(values) else 0.0
            self._command_at = time.monotonic()
            self._condition.notify_all()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        """Drop the target velocity; the worker stops on its next boundary."""
        with self._condition:
            self._raise_if_closed_locked()
            self._velocity = (0.0, 0.0, 0.0)
            self._ttl_s = 0.0
            self._command_at = time.monotonic()
            self._condition.notify_all()
        return self.snapshot()

    def set_recording(
        self,
        *,
        video: bool | None = None,
        actions: bool | None = None,
    ) -> dict[str, Any]:
        """Change what the rest of this run records, flushing video when it ends."""
        with self._condition:
            self._raise_if_closed_locked()
            previous = self._recording
            self._recording = replace(
                previous,
                video=previous.video if video is None else bool(video),
                actions=previous.actions if actions is None else bool(actions),
            )
            current = self._recording
        if previous.video and not current.video:
            self._flush_video()
        if previous.actions != current.actions:
            if current.actions:
                self._open_action_log()
            else:
                self._close_action_log()
        return self.snapshot()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> dict[str, Any]:
        """Stop driving, then write what the run recorded."""
        with self._condition:
            if not self._closed:
                self._closed = True
                self._velocity = (0.0, 0.0, 0.0)
                self._ttl_s = 0.0
                self._condition.notify_all()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.warning("teleop worker did not stop within 10s")
        try:
            self._env.stop()
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            logger.warning("teleop could not stop the base: %s", exc)
            with self._condition:
                self._error = self._error or f"stop failed: {exc}"
        self._flush_video()
        self._close_action_log()
        return self.summary()

    def snapshot(self) -> dict[str, Any]:
        """Return the dashboard projection of the current manual run."""
        with self._condition:
            return {
                "active": not self._closed,
                "velocity": list(self._velocity),
                "commanded_velocity": list(self._commanded),
                "recording": self._recording.as_dict(),
                "velocity_limits": list(self._limits),
                "main_frame": self._frame_key,
                "steps": self._steps,
                "clips": list(self._clips),
                "error": self._error,
            }

    def summary(self) -> dict[str, Any]:
        """Return what the finished run produced, for the TaskRun transcript."""
        with self._condition:
            return {
                "steps": self._steps,
                "clips": list(self._clips),
                "recording": self._recording.as_dict(),
                "error": self._error,
            }

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            velocity, deadline = self._next_command()
            if velocity is None:
                return
            duration = min(self._chunk_s, deadline - time.monotonic())
            if duration <= 0:
                continue
            try:
                self._drive_once(velocity, max(0.05, duration))
            except InteractionUnavailableError:
                return
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                logger.warning("teleop drive failed: %s", exc)
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._velocity = (0.0, 0.0, 0.0)
                    self._ttl_s = 0.0
                    self._condition.notify_all()

    def _next_command(self) -> tuple[tuple[float, float, float] | None, float]:
        """Wait for a live command; None means the worker must exit."""
        with self._condition:
            while not self._closed:
                now = time.monotonic()
                deadline = self._command_at + self._ttl_s
                if any(self._velocity):
                    if now < deadline:
                        return self._velocity, deadline
                    # The caller stopped refreshing: drop the target and idle
                    # until hold() or stop() supplies a new one.
                    self._velocity = (0.0, 0.0, 0.0)
                    self._ttl_s = 0.0
                self._condition.wait()
        return None, 0.0

    def _drive_once(
        self, velocity: tuple[float, float, float], duration_s: float
    ) -> None:
        """Drive one chunk, then publish its frame and action record."""
        started = time.monotonic()
        result = self._env.drive(
            np.asarray(velocity, dtype=np.float32), duration_s=duration_s
        )
        elapsed_s = round(time.monotonic() - started, 2)
        try:
            observation = self._env.get_observation()
        except Exception as exc:  # noqa: BLE001 - a lost frame must not stop driving
            logger.warning("teleop could not read a live frame: %s", exc)
            observation = {}
        frame = (
            observation.get("main_images") if isinstance(observation, dict) else None
        )

        with self._condition:
            self._steps += 1
            self._commanded = self._coerce_velocity(
                result.get("commanded_velocity", velocity)
            )
            record_video = self._recording.video
            record_actions = self._recording.actions

        if frame is not None:
            array = np.asarray(frame)
            self._publish_frame(array)
            if record_video:
                with self._condition:
                    self._frames.append(array)
        if record_actions:
            self._record_action(velocity, result, elapsed_s)

    # -- recording ---------------------------------------------------------

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Show one live frame in the Dashboard without persisting a step."""
        try:
            buffer = io.BytesIO()
            imageio.imwrite(buffer, np.ascontiguousarray(frame), format="png")
        except Exception as exc:  # noqa: BLE001 - the live view is best effort
            logger.warning("teleop could not encode a frame: %s", exc)
            return
        self._dashboard.publish_live_frames({self._frame_key: buffer.getvalue()})

    def _record_action(
        self,
        velocity: tuple[float, float, float],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        """Append one driven chunk to the action log and the timeline."""
        entry = {
            "t": round(time.monotonic() - self._started_at, 2),
            "velocity": list(velocity),
            "commanded_velocity": list(self._commanded),
            "duration_s": result.get("duration_s"),
            "steps": result.get("steps"),
        }
        with self._condition:
            handle = self._action_file
            if handle is not None:
                try:
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
                except Exception as exc:  # noqa: BLE001 - recording is best effort
                    logger.warning("teleop could not append to the action log: %s", exc)
                    self._action_file = None
        self._dashboard.record_teleop_action(
            {
                "action": "drive",
                "args": {
                    "velocity": entry["velocity"],
                    "duration_s": entry["duration_s"],
                },
                "result": {
                    "commanded_velocity": entry["commanded_velocity"],
                    "steps": entry["steps"],
                },
                "elapsed_s": elapsed_s,
                "terminated": False,
                "truncated": False,
                "has_action_video": False,
            }
        )

    def _flush_video(self) -> None:
        """Write the frames collected so far as one clip."""
        with self._condition:
            frames = self._frames
            self._frames = []
            index = len(self._clips)
        if not frames:
            return
        fps = max(1, int(round(1.0 / self._chunk_s)))
        name = f"teleop_{index:03d}.mp4"
        saved = None
        if self._env_state is not None:
            saved = self._env_state.save(name, frames, step=None, fps=fps)
        if saved is None:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    self._output_dir / name, frames, fps=fps, codec="libx264"
                )
                saved = name
            except Exception as exc:  # noqa: BLE001 - recording is best effort
                logger.warning("teleop could not write %s: %s", name, exc)
                with self._condition:
                    self._error = self._error or f"video recording failed: {exc}"
                return
        with self._condition:
            self._clips.append({"name": name, "frames": len(frames), "fps": fps})

    def _open_action_log(self) -> None:
        with self._condition:
            if not self._recording.actions or self._action_file is not None:
                return
            if self._closed:
                return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        handle = open(self._output_dir / ACTION_LOG_NAME, "a", encoding="utf-8")
        with self._condition:
            if not self._recording.actions or self._closed:
                handle.close()
                return
            self._action_file = handle

    def _close_action_log(self) -> None:
        with self._condition:
            handle = self._action_file
            self._action_file = None
        if handle is not None:
            try:
                handle.close()
            except OSError as exc:
                logger.warning("teleop could not close the action log: %s", exc)

    # -- helpers -----------------------------------------------------------

    def _read_environment(self) -> None:
        """Take the velocity limits and main camera name from the env."""
        try:
            limits = list(self._env.get_env_meta().get("velocity_limits") or ())
            if len(limits) == 3 and all(float(value) > 0 for value in limits):
                self._limits = tuple(abs(float(value)) for value in limits)
        except Exception as exc:  # noqa: BLE001 - fall back to defaults
            logger.warning("teleop could not read the env limits: %s", exc)
        try:
            cameras = self._env.get_camera_meta().get("observation_camera_map") or {}
            key = cameras.get("main")
            if isinstance(key, str) and key:
                self._frame_key = key
        except Exception as exc:  # noqa: BLE001 - fall back to the default name
            logger.warning("teleop could not read the camera map: %s", exc)

    @staticmethod
    def _coerce_velocity(velocity: Any) -> tuple[float, float, float]:
        array = np.asarray(velocity, dtype=np.float64)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError("velocity must contain exactly 3 finite numbers")
        return (float(array[0]), float(array[1]), float(array[2]))

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise InteractionUnavailableError("manual control has ended")
