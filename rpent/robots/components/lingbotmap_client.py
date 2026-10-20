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

"""Transport-independent client for RPent's LingBot-Map mapping service."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.utils.rpc import RpcClient

#: The map artifact returned by ``build`` is a binary blob; callers persist it
#: through ``EnvState`` rather than decoding it here.
DEFAULT_BUILD_TIMEOUT_S = 600.0
DEFAULT_PUSH_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class MapSessionStatus:
    """Current state of one LingBot-Map session."""

    session_id: str
    running: bool
    frames: int


@dataclass(frozen=True)
class MapBuildResult:
    """A finalized map artifact and its provenance."""

    map_path: str
    frames: int
    metadata: dict[str, Any]


class LingBotMapClient:
    """Client wrapping the LingBot-Map service over any :class:`RpcClient`.

    One client serializes one mapping session. Sessions are single-shot:
    ``start_session`` after ``build`` opens a fresh session id.
    """

    def __init__(self, client: RpcClient, *, timeout_s: float = 60.0) -> None:
        self._client = client
        self._timeout_s = timeout_s
        self._session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        """Return the active session id, or ``None`` before ``start_session``."""
        return self._session_id

    def start_session(self, *, map_name: str) -> MapSessionStatus:
        """Open a mapping session; refuse while a previous one is unbuilt."""
        if self._session_id is not None:
            raise RuntimeError(
                "a mapping session is already active; call build or stop first"
            )
        name = str(map_name).strip()
        if not name:
            raise ValueError("map_name must be a non-empty string")
        payload = self._client.call(
            "map.start_session",
            kwargs={"map_name": name},
            timeout_s=self._timeout_s,
        )
        status = self._decode_status(payload)
        self._session_id = status.session_id
        return status

    def push_frame(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        *,
        pose: list[float] | None,
        timestamp_s: float,
    ) -> MapSessionStatus:
        """Push one synchronized RGB frame and the robot pose that took it.

        ``pose`` is the robot base pose in the map frame as a flat 6-vector
        ``[x, y, yaw, roll, pitch, yaw_head]``, or ``None`` when the robot
        exposes no odometry and the mapping service estimates pose from frames
        alone.
        """
        if self._session_id is None:
            raise RuntimeError("no active mapping session; call start_session first")
        if pose is not None:
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose_array.shape != (6,):
                raise ValueError(
                    "pose must be a flat 6-vector [x, y, yaw, roll, pitch, yaw_head]"
                )
            if not np.isfinite(pose_array).all():
                raise ValueError("pose must contain only finite values")
            pose_value: list[float] | None = pose_array.tolist()
        else:
            pose_value = None
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if isinstance(image, np.ndarray):
            buffer = io.BytesIO()
            imageio.imwrite(buffer, image, format="png")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = bytes(image)
        if not image_bytes:
            raise ValueError("image is empty")
        payload = self._client.call(
            "map.push_frame",
            kwargs={
                "session_id": self._session_id,
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "pose": pose_value,
                "timestamp_s": timestamp,
            },
            timeout_s=DEFAULT_PUSH_TIMEOUT_S,
        )
        return self._decode_status(payload)

    def build(self, *, timeout_s: float = DEFAULT_BUILD_TIMEOUT_S) -> MapBuildResult:
        """Finalize the active session and save the map artifact."""
        if self._session_id is None:
            raise RuntimeError("no active mapping session; call start_session first")
        payload = self._client.call(
            "map.build",
            kwargs={"session_id": self._session_id},
            timeout_s=timeout_s,
        )
        self._session_id = None
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid map.build response: {payload!r}")
        map_path = payload.get("map_path")
        frames = payload.get("frames")
        metadata = payload.get("metadata", {})
        if not isinstance(map_path, str) or not map_path:
            raise RuntimeError(f"map.build response omitted map_path: {payload!r}")
        if not isinstance(frames, int) or frames < 0:
            raise RuntimeError(f"map.build response has invalid frames: {payload!r}")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"map.build response has invalid metadata: {payload!r}")
        return MapBuildResult(
            map_path=map_path,
            frames=frames,
            metadata=dict(metadata),
        )

    def stop_session(self) -> MapSessionStatus | None:
        """Discard the active session without building a map; idempotent."""
        if self._session_id is None:
            return None
        session_id, self._session_id = self._session_id, None
        payload = self._client.call(
            "map.stop_session",
            kwargs={"session_id": session_id},
            timeout_s=self._timeout_s,
        )
        return self._decode_status(payload)

    def load_map_metadata(self, map_path: str) -> dict[str, Any]:
        """Read a saved map's metadata JSON without loading the map itself."""
        payload = self._client.call(
            "map.load_metadata",
            kwargs={"map_path": str(map_path)},
            timeout_s=self._timeout_s,
        )
        if not isinstance(payload, dict) or "metadata" not in payload:
            raise RuntimeError(f"invalid map.load_metadata response: {payload!r}")
        return payload["metadata"]

    @staticmethod
    def _decode_status(payload: Any) -> MapSessionStatus:
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid mapping session response: {payload!r}")
        session_id = payload.get("session_id")
        running = payload.get("running")
        frames = payload.get("frames")
        if not isinstance(session_id, str) or not isinstance(running, bool):
            raise RuntimeError(f"invalid mapping session response: {payload!r}")
        if not isinstance(frames, int) or frames < 0:
            raise RuntimeError(f"invalid session frames response: {payload!r}")
        return MapSessionStatus(
            session_id=session_id,
            running=running,
            frames=frames,
        )
