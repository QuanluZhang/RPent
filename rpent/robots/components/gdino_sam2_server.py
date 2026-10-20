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

"""RPC server owning the local Grounding DINO + SAM 2 perception models.

Run manually with::

    GDINO_CHECKPOINT_PATH=/path/to/groundingdino_swint_ogc.pth \\
    SAM2_CHECKPOINT_PATH=/path/to/sam2.1_hiera_large.pt \\
        python -m rpent.robots.components.gdino_sam2_server \\
        --transport http --host 127.0.0.1 --port 8115

RPent normally starts this process automatically. The service exposes one
``gdino_sam2.detect_segment`` RPC method over either HTTP or socket transport:
Grounding DINO produces open-vocabulary boxes for each label, and SAM 2
refines each box into an instance mask on the same frame.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("gdino_sam2_server")

MAX_LABELS_PER_CALL = 16


def _encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class GdinoSam2Facade(RpcFacade):
    """RPC server wrapping Grounding DINO detection and SAM 2 segmentation."""

    def __init__(self, gdino_checkpoint: str, sam2_checkpoint: str) -> None:
        super().__init__()
        self._load(gdino_checkpoint, sam2_checkpoint)
        self._register_rpc()

    def _load(self, gdino_checkpoint: str, sam2_checkpoint: str) -> None:
        try:
            import torch
            from groundingdino.util.inference import load_model as load_gdino
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise RuntimeError(
                "local Grounding DINO + SAM 2 dependencies are missing; install "
                'RPent with pip install -e ".[grounding-sam2]"'
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("local Grounding DINO + SAM 2 requires a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)

        self._gdino_checkpoint = _resolve_checkpoint(
            gdino_checkpoint, "GDINO_CHECKPOINT_PATH"
        )
        self._sam2_checkpoint = _resolve_checkpoint(
            sam2_checkpoint, "SAM2_CHECKPOINT_PATH"
        )
        logger.info("loading Grounding DINO: %s", self._gdino_checkpoint)
        self._gdino = load_gdino(self._gdino_checkpoint)
        logger.info("loading SAM 2: %s", self._sam2_checkpoint)
        self._sam2 = build_sam2(
            config_file=self._sam2_config(),
            ckpt_path=self._sam2_checkpoint,
            device="cuda",
        )
        self._torch = torch
        self._device = "cuda"
        self._lock = threading.Lock()

    def _sam2_config(self) -> str:
        """SAM 2 model config matching the checkpoint, from its filename."""
        name = Path(self._sam2_checkpoint).name.lower()
        for key in ("hiera_large", "hiera_base_plus", "hiera_small", "hiera_tiny"):
            if key in name:
                return f"configs/sam2.1/{key.replace('_', '_')}.yaml"
        return "configs/sam2.1/sam2.1_hiera_large.yaml"

    def _register_rpc(self) -> None:
        self._rpc["gdino_sam2.detect_segment"] = self.detect_segment
        self._readonly_methods.add("gdino_sam2.detect_segment")

    def _inference_context(self):
        if self._torch is None or not self._device.startswith("cuda"):
            return nullcontext()
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def detect_segment(
        self,
        *,
        image_base64: str,
        labels: list[str],
        min_score: float = 0.3,
    ) -> dict[str, Any]:
        """Detect each label with Grounding DINO; segment each box with SAM 2."""
        if not isinstance(image_base64, str) or not image_base64:
            raise ValueError("image_base64 must be a non-empty string")
        prompts = [str(label).strip() for label in labels or []]
        if not prompts or any(not label for label in prompts):
            raise ValueError("labels must be non-empty strings")
        if len(prompts) > MAX_LABELS_PER_CALL:
            raise ValueError(f"at most {MAX_LABELS_PER_CALL} labels per call")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_score must be a number, got {min_score!r}") from exc
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        try:
            raw = base64.b64decode(image_base64, validate=True)
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be RGB [H,W,3], got {image.shape}")

        with self._lock:
            detections = self._detect(image, prompts, min_score)
        return {
            "found": bool(detections),
            "detections": detections,
            "image_shape": [int(image.shape[0]), int(image.shape[1])],
        }

    def _detect(
        self, image: np.ndarray, prompts: list[str], min_score: float
    ) -> list[dict[str, Any]]:
        from groundingdino.util.inference import predict as predict_gdino

        torch = self._torch
        frame = torch.from_numpy(np.asarray(image)).to(self._device)
        frame = frame.permute(2, 0, 1).float() / 255.0
        detections: list[dict[str, Any]] = []
        with self._inference_context():
            boxes, scores, phrases = predict_gdino(
                model=self._gdino,
                image=frame,
                caption=". ".join(prompts) + ".",
                box_threshold=min_score,
                text_threshold=min_score,
            )
        if boxes is None or scores is None:
            return detections
        boxes_np = (
            boxes if isinstance(boxes, np.ndarray) else boxes.detach().float().cpu().numpy()
        )
        scores_np = (
            scores
            if isinstance(scores, np.ndarray)
            else scores.detach().float().cpu().numpy()
        ).reshape(-1)
        phrases = list(phrases or ())
        if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
            return detections
        height, width = image.shape[:2]
        keep: list[tuple[float, list[float], int]] = []
        for index, (box, score) in enumerate(zip(boxes_np, scores_np, strict=True)):
            label = phrases[index] if index < len(phrases) else ""
            if label not in prompts:
                continue
            keep.append((float(score), [float(value) for value in box], index))
        keep.sort(key=lambda item: -item[0])
        for score, cxcywh, index in keep:
            mask = self._segment(image, cxcywh)
            if mask is None:
                continue
            x1 = max(0.0, (cxcywh[0] - cxcywh[2] / 2) * width)
            y1 = max(0.0, (cxcywh[1] - cxcywh[3] / 2) * height)
            x2 = min(float(width), (cxcywh[0] + cxcywh[2] / 2) * width)
            y2 = min(float(height), (cxcywh[1] + cxcywh[3] / 2) * height)
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "box": [x1, y1, x2, y2],
                    "mask_png_base64": _encode_mask_png(mask),
                }
            )
        return detections

    def _segment(self, image: np.ndarray, cxcywh: list[float]) -> np.ndarray | None:
        """Refine one normalized cxcywh box into an instance mask with SAM 2."""
        torch = self._torch
        height, width = image.shape[:2]
        box = torch.tensor(
            [[
                max(0.0, (cxcywh[0] - cxcywh[2] / 2)) * width,
                max(0.0, (cxcywh[1] - cxcywh[3] / 2)) * height,
                min(float(width), (cxcywh[0] + cxcywh[2] / 2) * width),
                min(float(height), (cxcywh[1] + cxcywh[3] / 2) * height),
            ]],
            device=self._device,
        )
        if box[0][2] <= box[0][0] or box[0][3] <= box[0][1]:
            return None
        frame = torch.from_numpy(np.asarray(image)).to(self._device)
        frame = frame.permute(2, 0, 1).float()
        with self._inference_context():
            masks, scores, _ = self._sam2.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
        mask = masks[int(np.argmax(scores.detach().float().cpu().numpy()))]
        mask_np = (
            mask if isinstance(mask, np.ndarray) else mask.detach().float().cpu().numpy()
        )
        return np.asarray(mask_np) > 0


def _resolve_checkpoint(value: str, env_name: str) -> str:
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found ({env_name}): {resolved}")
    return str(resolved)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RPent Grounding DINO + SAM 2 server"
    )
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """Load both models and serve until terminated."""
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target
    gdino_checkpoint = os.environ.get("GDINO_CHECKPOINT_PATH")
    sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT_PATH")
    if not gdino_checkpoint or not sam2_checkpoint:
        raise RuntimeError(
            "GDINO_CHECKPOINT_PATH and SAM2_CHECKPOINT_PATH must be set; export "
            "both checkpoint paths before starting RPent"
        )
    facade = GdinoSam2Facade(gdino_checkpoint, sam2_checkpoint)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
