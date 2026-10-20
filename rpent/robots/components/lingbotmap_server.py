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

"""RPC server owning a LingBot-Map workspace mapping session.

Run manually with::

    LINGBOTMAP_OUTPUT_DIR=/path/to/maps \\
        python -m rpent.robots.components.lingbotmap_server \\
        --transport http --host 127.0.0.1 --port 8116

RPent normally starts this process automatically. The service exposes
``map.start_session`` / ``map.push_frame`` / ``map.build`` / ``map.stop_session``
/ ``map.load_metadata`` RPC methods over either HTTP or socket transport.

The LingBot-Map builder is wrapped behind the small :class:`MapBackend`
protocol so the concrete integration stays in one adapter function. The
default backend (:class:`FrameArchiveBackend`) records frames and poses and
serializes them as the map artifact, which keeps the RPC contract and the
tool flow verifiable offline.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("lingbotmap_server")

DEFAULT_OUTPUT_DIR = "/tmp/rpent_lingbotmap"


class MapBackend(Protocol):
    """The slice of LingBot-Map the mapping tool depends on."""

    def start(self, map_name: str) -> None:
        """Open a new mapping run named ``map_name``."""

    def push_frame(
        self, image: np.ndarray, pose: np.ndarray | None, timestamp_s: float
    ) -> None:
        """Record one synchronized RGB frame and base pose."""

    def build(self) -> dict[str, Any]:
        """Finalize the map; return artifact metadata including ``map_path``."""


class FrameArchiveBackend:
    """Reference backend that archives frames and poses as the map artifact.

    The archive is a directory with ``frames/*.png``, ``poses.json``, and a
    ``map.json`` metadata file. It exists so the mapping RPC contract and the
    ``build_map`` tool flow run without LingBot-Map installed; a real
    LingBot-Map adapter implements :class:`MapBackend` instead.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._map_name = ""
        self._frames: list[np.ndarray] = []
        self._poses: list[list[float] | None] = []
        self._timestamps: list[float] = []

    def start(self, map_name: str) -> None:
        self._map_name = map_name
        self._frames = []
        self._poses = []
        self._timestamps = []

    def push_frame(
        self, image: np.ndarray, pose: np.ndarray | None, timestamp_s: float
    ) -> None:
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"pushed frame must be RGB [H,W,3], got {frame.shape}")
        self._frames.append(frame)
        self._poses.append(
            np.asarray(pose, dtype=np.float64).tolist() if pose is not None else None
        )
        self._timestamps.append(float(timestamp_s))

    def build(self) -> dict[str, Any]:
        if not self._map_name:
            raise RuntimeError("no mapping run started")
        map_dir = self._output_dir / self._map_name
        frames_dir = map_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(self._frames):
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                frames_dir / f"{index:06d}.png"
            )
        poses = [
            {"pose": pose, "timestamp_s": timestamp}
            for pose, timestamp in zip(self._poses, self._timestamps, strict=True)
        ]
        (map_dir / "poses.json").write_text(json.dumps(poses))
        metadata = {
            "backend": "frame_archive",
            "map_name": self._map_name,
            "frames": len(self._frames),
        }
        (map_dir / "map.json").write_text(json.dumps(metadata))
        logger.info("built map %s with %d frames", self._map_name, len(self._frames))
        return {
            "map_path": str(map_dir),
            "frames": len(self._frames),
            "metadata": metadata,
        }


def build_lingbot_backend(output_dir: str) -> MapBackend:
    """Resolve the LingBot-Map backend for this server run.

    The reference archive backend is the default. When the ``lingbotmap``
    import is available, it is used instead; its exact constructor is fixed
    once the concrete LingBot-Map package API is pinned.
    """
    try:
        import lingbotmap  # type: ignore[import-not-found]

        logger.info("using lingbotmap package %s", getattr(lingbotmap, "__version__", "?"))
        return FrameArchiveBackend(output_dir)  # placeholder until API pinned
    except ImportError:
        logger.info("lingbotmap not installed; using frame-archive backend")
        return FrameArchiveBackend(output_dir)


class LingBotMapFacade(RpcFacade):
    """RPC server owning one LingBot-Map mapping session at a time."""

    def __init__(self, backend: MapBackend) -> None:
        super().__init__()
        self._backend = backend
        self._session_id: str | None = None
        self._session_frames = 0
        self._register_rpc()

    def _register_rpc(self) -> None:
        self._rpc["map.start_session"] = self.start_session
        self._rpc["map.push_frame"] = self.push_frame
        self._rpc["map.build"] = self.build
        self._rpc["map.stop_session"] = self.stop_session
        self._rpc["map.load_metadata"] = self.load_metadata

    # ---- RPC methods -----------------------------------------------------

    def start_session(self, *, map_name: str) -> dict[str, Any]:
        name = str(map_name).strip()
        if not name:
            raise ValueError("map_name must be a non-empty string")
        if self._session_id is not None:
            raise RuntimeError(
                "a mapping session is already active; build or stop it first"
            )
        self._backend.start(name)
        self._session_id = uuid.uuid4().hex[:12]
        self._session_frames = 0
        return self._status()

    def push_frame(
        self,
        *,
        session_id: str,
        image_base64: str,
        pose: list[float] | None,
        timestamp_s: float,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("image_base64 must be a non-empty string")
        try:
            raw = base64.b64decode(image_base64, validate=True)
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc
        if pose is not None:
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose_array.shape != (6,) or not np.isfinite(pose_array).all():
                raise ValueError("pose must be a finite flat 6-vector")
        else:
            pose_array = None
        self._backend.push_frame(image, pose_array, float(timestamp_s))
        self._session_frames += 1
        return self._status()

    def build(self, *, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        if self._session_frames == 0:
            self._session_id = None
            raise RuntimeError("no frames were pushed; refusing to build an empty map")
        try:
            result = self._backend.build()
        except BaseException:
            self._session_id = None
            raise
        self._session_id = None
        return result

    def stop_session(self, *, session_id: str) -> dict[str, Any]:
        self._require_session(session_id)
        self._session_id = None
        self._session_frames = 0
        return self._status()

    def load_metadata(self, *, map_path: str) -> dict[str, Any]:
        metadata_path = Path(map_path) / "map.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"map metadata not found: {metadata_path}")
        return {"metadata": json.loads(metadata_path.read_text())}

    # ---- helpers ---------------------------------------------------------

    def _require_session(self, session_id: str) -> None:
        if not isinstance(session_id, str) or session_id != self._session_id:
            raise RuntimeError("unknown or expired mapping session")

    def _status(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id or "",
            "running": self._session_id is not None,
            "frames": self._session_frames,
        }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent LingBot-Map server")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for built map artifacts (default: LINGBOTMAP_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Start the mapping facade and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    output_dir = args.output_dir or os.environ.get("LINGBOTMAP_OUTPUT_DIR")
    if not output_dir:
        raise RuntimeError(
            "map output directory is not set; pass --output-dir or export "
            "LINGBOTMAP_OUTPUT_DIR before starting RPent"
        )
    facade = LingBotMapFacade(build_lingbot_backend(output_dir))
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
