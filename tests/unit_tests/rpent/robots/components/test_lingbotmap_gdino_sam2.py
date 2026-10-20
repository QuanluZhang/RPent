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

"""Offline tests for the LingBot-Map and Grounding DINO + SAM 2 services."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rpent.robots.components.gdino_sam2_client import (
    GdinoSam2Client,
)
from rpent.robots.components.lingbotmap_client import LingBotMapClient
from rpent.robots.components.lingbotmap_server import (
    FrameArchiveBackend,
    LingBotMapFacade,
)
from rpent.utils.rpc import RpcClient


class _LocalRpcClient(RpcClient):
    """Dispatch calls directly into a facade's registered RPC methods."""

    def __init__(self, facade: Any) -> None:
        self._facade = facade

    def call(self, method: str, args: tuple = (), kwargs: dict | None = None, **_: Any) -> Any:
        handler = getattr(self._facade, "_rpc")[method]
        return handler(*(args or ()), **(kwargs or {}))


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _frame(fill: int = 128) -> np.ndarray:
    return np.full((24, 32, 3), fill, dtype=np.uint8)


@pytest.fixture
def map_client(tmp_path: Path) -> LingBotMapClient:
    facade = LingBotMapFacade(FrameArchiveBackend(str(tmp_path / "maps")))
    return LingBotMapClient(_LocalRpcClient(facade))


class TestLingBotMapService:
    def test_start_push_build_roundtrip(self, map_client: LingBotMapClient) -> None:
        status = map_client.start_session(map_name="office")
        assert status.running and status.frames == 0

        for index in range(3):
            pushed = map_client.push_frame(
                _frame(index * 30), pose=[0.0, 0.0, 0.1 * index, 0, 0, 0],
                timestamp_s=1.0 + index,
            )
            assert pushed.frames == index + 1

        built = map_client.build()
        assert built.frames == 3
        assert (Path(built.map_path) / "map.json").is_file()
        assert len(list((Path(built.map_path) / "frames").glob("*.png"))) == 3
        assert map_client.session_id is None

    def test_push_without_pose_estimates_from_frames(
        self, map_client: LingBotMapClient, tmp_path: Path
    ) -> None:
        map_client.start_session(map_name="no_odom")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        import json

        poses = json.loads((Path(built.map_path) / "poses.json").read_text())
        assert poses[0]["pose"] is None

    def test_build_refuses_empty_session(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="empty")
        with pytest.raises(RuntimeError, match="no frames"):
            map_client.build()

    def test_push_frame_rejects_bad_pose(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="bad_pose")
        with pytest.raises(ValueError, match="flat 6-vector"):
            map_client.push_frame(_frame(), pose=[1.0, 2.0], timestamp_s=1.0)
        with pytest.raises(ValueError, match="finite"):
            map_client.push_frame(
                _frame(), pose=[float("nan")] * 6, timestamp_s=1.0
            )

    def test_client_blocks_second_session_until_resolved(
        self, map_client: LingBotMapClient
    ) -> None:
        map_client.start_session(map_name="one")
        with pytest.raises(RuntimeError, match="already active"):
            map_client.start_session(map_name="two")
        assert map_client.session_id is not None
        map_client.stop_session()
        assert map_client.session_id is None
        status = map_client.start_session(map_name="two")
        assert status.running

    def test_push_without_session_is_refused(self, map_client: LingBotMapClient) -> None:
        with pytest.raises(RuntimeError, match="no active mapping session"):
            map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)

    def test_load_metadata_reads_built_map(self, map_client: LingBotMapClient) -> None:
        map_client.start_session(map_name="meta")
        map_client.push_frame(_frame(), pose=None, timestamp_s=1.0)
        built = map_client.build()
        metadata = map_client.load_map_metadata(built.map_path)
        assert metadata["map_name"] == "meta"
        assert metadata["frames"] == 1


class _FakePerceptionFacade:
    """Deterministic perception backend shaped like the real RPC response."""

    _rpc = {"gdino_sam2.detect_segment": None}  # replaced per instance below

    def __init__(self) -> None:
        self._rpc = {"gdino_sam2.detect_segment": self.detect_segment}

    def detect_segment(
        self,
        *,
        image_base64: str,
        labels: list[str],
        min_score: float = 0.3,
    ) -> dict[str, Any]:
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(image_base64))))
        mask = image[..., 0] > 100
        detections = [
            {
                "label": label,
                "score": 0.9,
                "box": [2.0, 2.0, 30.0, 20.0],
                "mask_png_base64": base64.b64encode(_png_bytes(mask * 255)).decode(),
            }
            for label in labels
        ]
        return {
            "found": bool(detections),
            "detections": detections,
            "image_shape": [image.shape[0], image.shape[1]],
        }


class TestGdinoSam2Client:
    def test_detect_segment_decodes_masks(self) -> None:
        client = GdinoSam2Client(_LocalRpcClient(_FakePerceptionFacade()))
        result = client.detect_segment(_frame(200), labels=["chair", "table"])
        assert result.found
        assert [d.label for d in result.detections] == ["chair", "table"]
        assert result.image_shape == (24, 32)
        for detection in result.detections:
            assert detection.mask.shape == (24, 32)
            assert detection.mask.sum() == 24 * 32

    def test_client_validates_labels(self) -> None:
        client = GdinoSam2Client(_LocalRpcClient(_FakePerceptionFacade()))
        with pytest.raises(ValueError, match="non-empty strings"):
            client.detect_segment(_frame(), labels=[])
        with pytest.raises(ValueError, match="non-empty strings"):
            client.detect_segment(_frame(), labels=["  "])
        with pytest.raises(ValueError, match="at most 16"):
            client.detect_segment(_frame(), labels=[f"obj{i}" for i in range(17)])
        with pytest.raises(ValueError, match="min_score"):
            client.detect_segment(_frame(), labels=["chair"], min_score=1.5)
