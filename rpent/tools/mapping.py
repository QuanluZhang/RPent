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

"""Robot-agnostic workspace mapping tools.

:class:`MappingMixin` turns any :class:`~rpent.tools.toolkit.Toolkit` whose
robot can serve a synchronized RGB frame and a base pose into a mapping agent:

1. ``build_map`` opens a LingBot-Map session, asks the operator to teleoperate
   the robot around the workspace while a poller thread pushes frames and
   poses, finalizes the map when the operator replies, optionally labels
   requested objects on keyframes with Grounding DINO + SAM 2, and persists the
   map bundle as ``EnvState`` artifacts.
2. ``detect_objects`` runs Grounding DINO + SAM 2 standalone on any persisted
   camera frame, so the perception service stays usable outside mapping.

The robot-specific slice is two methods the mixin calls: ``_map_frame()``
returns ``(rgb, pose, timestamp)`` for the current sensor read, and the
toolkit's ``get_env_state`` keeps recording steps as usual while the operator
drives.
"""

from __future__ import annotations

import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import Field

from rpent.robots.components.gdino_sam2_client import GdinoSam2Client
from rpent.robots.components.lingbotmap_client import LingBotMapClient
from rpent.tools import ToolResult, tool
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.tools.base import Tool
    from rpent.tools.toolkit import Toolkit

logger = get_logger("mapping")

#: Default sampling cadence for frames pushed during teleoperation.
DEFAULT_FRAME_INTERVAL_S = 0.5
#: Minimum fraction of pushed frames kept as labeling keyframes.
KEYFRAME_STRIDE = 20


class MapFrameSource(Protocol):
    """The robot-specific sensor read the mapping mixin depends on."""

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        """Return ``(rgb, base_pose, timestamp_s)`` for the current read.

        ``rgb`` is an RGB ``[H,W,3]`` array or PNG bytes. ``base_pose`` is the
        robot base pose in the map frame as a flat 6-vector
        ``[x, y, yaw, roll, pitch, yaw_head]``, or ``None`` when the robot
        does not expose odometry and the mapping service must estimate pose
        from frames alone.
        """


class MappingMixin:
    """Teleop-driven map construction for any robot toolkit.

    Hosts provide ``self._lingbotmap_client``, and optionally
    ``self._gdino_sam2_client``, ``self._operator_input`` (a
    ``HumanInTheLoopInput`` or ``OperatorReader``), plus ``_map_frame()``.
    Register the tools with :meth:`register_mapping_tools` after the toolkit
    holds its primitives.
    """

    _frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S

    def register_mapping_tools(self) -> None:
        """Register ``build_map`` and, with a perception client, ``detect_objects``."""
        self.add_tool(self.build_map)  # type: ignore[attr-defined]
        if getattr(self, "_gdino_sam2_client", None) is not None:
            definition: Tool = self.detect_objects  # type: ignore[attr-defined]
            self.add_tool(
                definition.with_handler(
                    partial(definition, state=self.state)  # type: ignore[attr-defined]
                )
            )

    # ---- robot-provided hooks --------------------------------------------

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        """Return the current synchronized ``(rgb, base_pose, timestamp_s)``."""
        raise NotImplementedError("the robot toolkit must provide _map_frame")

    def _ask_operator(self, prompt: str, *, kind: str | None = None) -> str | None:
        """Wait for the operator's reply, or ``None`` when no channel exists."""
        operator_input = getattr(self, "_operator_input", None)  # type: ignore[attr-defined]
        if operator_input is not None:
            request = getattr(operator_input, "request", None)
            if callable(request):
                return request(prompt, self.raise_if_cancelled, kind=kind)  # type: ignore[attr-defined]
            return operator_input(prompt, self.raise_if_cancelled)  # type: ignore[attr-defined]
        # No operator channel was wired in (robots without human-interactive
        # exploration); a plain TTY still lets a terminal operator reply.
        import sys

        if sys.stdin is None or not sys.stdin.isatty():
            return None
        try:
            return input(f"{prompt}\nReply: ").strip()
        except EOFError:
            return None

    # ---- tools -------------------------------------------------------------

    @tool
    def build_map(
        self,
        *,
        map_name: str,
        labels: Annotated[
            list[str],
            Field(
                max_length=16,
                json_schema_extra={"default": []},
            ),
        ] = [],
    ) -> ToolResult:
        """Build a workspace map by having the operator teleoperate the robot.

        Opens a LingBot-Map mapping session, then asks the operator to
        teleoperate the robot around the workspace; frames and base poses are
        pushed while they drive. When the operator replies, the map is built
        and saved. When labels are given, each is detected and segmented on
        saved keyframes with Grounding DINO + SAM 2 and annotated onto the map.

        Args:
            map_name: Name for the saved map; also names the map artifacts.
            labels: Optional object labels (at most 16) to detect on keyframes
                and annotate onto the map with Grounding DINO + SAM 2.
        """
        map_client: LingBotMapClient = self._require_map_client()  # type: ignore[attr-defined]
        name = str(map_name).strip()
        if not name:
            return ToolResult(data={"error": "map_name must be a non-empty string"})
        client_labels = [str(label).strip() for label in labels or []]
        if any(not label for label in client_labels):
            return ToolResult(data={"error": "labels must be non-empty strings"})
        if client_labels and getattr(self, "_gdino_sam2_client", None) is None:
            return ToolResult(
                data={
                    "error": "labels require the Grounding DINO + SAM 2 service; "
                    "restart with --gdino-sam2-endpoint or omit labels"
                }
            )

        started = time.perf_counter()
        try:
            status = map_client.start_session(map_name=name)
        except Exception as exc:
            return ToolResult(data={"error": f"could not start mapping session: {exc}"})

        poller = _FramePoller(
            source=self,  # type: ignore[arg-type]
            map_client=map_client,
            interval_s=self._frame_interval_s,
        )
        try:
            poller.start()
            reply = self._ask_operator(
                "Mapping session is open. Teleoperate the robot slowly around the "
                "workspace now; reply 'done' when the sweep is complete.",
                kind="reset",
            )
        except BaseException:
            poller.stop()
            self._stop_mapping_session(map_client)
            raise
        pushed = poller.stop()
        if pushed == 0:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": "no frames reached the mapping service",
                    "frames_pushed": 0,
                }
            )
        if reply is None:
            self._stop_mapping_session(map_client)
            return ToolResult(
                data={
                    "error": (
                        "no operator reply arrived; the teleoperation sweep was not "
                        "confirmed, so the map was discarded"
                    ),
                    "frames_pushed": pushed,
                }
            )

        try:
            build = map_client.build()
        except Exception as exc:
            return ToolResult(data={"error": f"map build failed: {exc}"})

        annotations: dict[str, Any] = {}
        if client_labels:
            annotations = self._label_keyframes(
                poller.keyframes, client_labels
            )
        elapsed_s = round(time.perf_counter() - started, 2)
        return ToolResult(
            data={
                "success": True,
                "map_name": name,
                "session_id": status.session_id,
                "frames_pushed": pushed,
                "operator_reply": reply,
                "map_path": build.map_path,
                "map_metadata": build.metadata,
                "labels": annotations,
                "elapsed_s": elapsed_s,
            }
        )

    @tool(readonly=True, exclude=("state",))
    def detect_objects(
        self,
        *,
        labels: Annotated[list[str], Field(min_length=1, max_length=16)],
        step: Annotated[
            int, Field(json_schema_extra={"default": -1})
        ] = -1,
        min_score: Annotated[
            float, Field(ge=0.0, le=1.0, json_schema_extra={"default": 0.3})
        ] = 0.3,
        state: Any = None,
    ) -> ToolResult:
        """Detect and segment named objects on one persisted camera frame.

        Runs Grounding DINO + SAM 2 on the saved RGB frame at ``step``
        (default: latest). Returns each detection's label, score, box, pixel
        count, and the artifact path of its rendered mask overlay.
        """
        return _detect_objects_impl(
            self._require_perception_client(),  # type: ignore[attr-defined]
            state,
            labels=labels,
            step=step,
            min_score=min_score,
        )

    # ---- helpers -----------------------------------------------------------

    def _require_map_client(self) -> LingBotMapClient:
        client = getattr(self, "_lingbotmap_client", None)
        if client is None:
            raise RuntimeError("no LingBot-Map service is connected")
        return client

    def _require_perception_client(self) -> GdinoSam2Client:
        client = getattr(self, "_gdino_sam2_client", None)
        if client is None:
            raise RuntimeError("no Grounding DINO + SAM 2 service is connected")
        return client

    def _stop_mapping_session(self, map_client: LingBotMapClient) -> None:
        try:
            map_client.stop_session()
        except Exception as exc:
            logger.warning("failed to stop mapping session: %s", exc)

    def _label_keyframes(
        self,
        keyframes: list[tuple[Any, list[float] | None, float]],
        labels: list[str],
    ) -> dict[str, Any]:
        """Detect each label on saved keyframes and fuse annotations by pose.

        The first frame's base pose (or the map origin when no pose exists)
        anchors each annotation: the detected box's pixel centroid is reported
        alongside the robot pose that observed it, so later consumers can
        triangulate world positions from the persisted keyframes.
        """
        perception = self._require_perception_client()
        annotations: dict[str, Any] = {}
        for label in labels:
            best: dict[str, Any] | None = None
            for index, (rgb, pose, timestamp) in enumerate(keyframes):
                result = perception.detect_segment(rgb, labels=[label])
                if not result.found:
                    continue
                top = max(result.detections, key=lambda item: item.score)
                candidate = {
                    "keyframe": index,
                    "score": top.score,
                    "box": top.box,
                    "pixel_count": int(top.mask.sum()),
                    "base_pose": pose,
                    "timestamp_s": timestamp,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
            if best is not None:
                annotations[label] = best
        return annotations


class _FramePoller:
    """Push frames and poses to a mapping session from a background thread."""

    def __init__(
        self,
        *,
        source: "Toolkit | MapFrameSource",
        map_client: LingBotMapClient,
        interval_s: float,
    ) -> None:
        self._source = source
        self._map_client = map_client
        self._interval_s = max(0.1, float(interval_s))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pushed = 0
        self._error: str | None = None
        self._lock = threading.Lock()
        #: Sampled keyframes kept for labeling: ``(rgb, pose, timestamp)``.
        self.keyframes: list[tuple[Any, list[float] | None, float]] = []

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="map-frame-poller", daemon=True
        )
        self._thread.start()

    def stop(self) -> int:
        """Signal the poller to stop, join it, and return pushed-frame count."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._error is not None:
            logger.warning("frame poller stopped early: %s", self._error)
        return self._pushed

    def _run(self) -> None:
        next_due = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_due:
                self._stop_event.wait(min(next_due - now, self._interval_s))
                continue
            next_due = now + self._interval_s
            try:
                rgb, pose, timestamp = self._source._map_frame()
                if pose is None:
                    status = self._map_client.push_frame(rgb, pose=None, timestamp_s=timestamp)
                else:
                    status = self._map_client.push_frame(
                        rgb, pose=pose, timestamp_s=timestamp
                    )
            except Exception as exc:
                self._error = str(exc)
                return
            with self._lock:
                self._pushed = status.frames
                if self._pushed % KEYFRAME_STRIDE == 1 or len(self.keyframes) < 1:
                    self.keyframes.append((rgb, pose, timestamp))


def _detect_objects_impl(
    perception: GdinoSam2Client,
    state: Any,
    *,
    labels: list[str],
    step: int,
    min_score: float,
) -> ToolResult:
    """Shared body for the ``detect_objects`` tool; kept out of the mixin MRO."""
    prompts = [str(label).strip() for label in labels]
    if not prompts or any(not label for label in prompts):
        return ToolResult(data={"error": "labels must be non-empty strings"})
    try:
        record = state.get(step)
    except Exception as exc:
        return ToolResult(data={"error": f"state step not available: {exc}"})
    image_name = next(
        (name for name in sorted(record.artifacts) if name.endswith(".png")),
        None,
    )
    if image_name is None:
        return ToolResult(
            data={
                "error": "no persisted RGB frame at this step",
                "step": record.step_idx,
                "artifacts": sorted(record.artifacts),
            }
        )
    image_bytes = state.load_bytes(image_name, step=record.step_idx)
    try:
        result = perception.detect_segment(image_bytes, labels=prompts, min_score=min_score)
    except Exception as exc:
        return ToolResult(data={"error": f"detection failed: {exc}"})
    if not result.found:
        return ToolResult(
            data={
                "success": True,
                "found": False,
                "step": record.step_idx,
                "labels": prompts,
                "reason": result.reason,
            }
        )
    return ToolResult(
        data={
            "success": True,
            "found": True,
            "step": record.step_idx,
            "image": image_name,
            "image_shape": list(result.image_shape),
            "detections": [
                {
                    "label": detection.label,
                    "score": round(detection.score, 4),
                    "box": [round(value, 1) for value in detection.box],
                    "pixel_count": int(detection.mask.sum()),
                }
                for detection in result.detections
            ],
        }
    )
