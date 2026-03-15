from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class ClickControllerOnnxNumpy:
    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",
        max_clicks: int = 8,
        click_radius: int = 5,
        with_flip: bool = True,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for the ONNX click backend."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        if len(self.input_names) < 2:
            raise RuntimeError("RITM ONNX must expose image and coord_features inputs.")
        self.image_input = self.input_names[0]
        self.coord_input = self.input_names[1]
        image_shape = self.session.get_inputs()[0].shape
        self.with_prev_mask = int(image_shape[1]) == 4
        self.max_clicks = int(max_clicks)
        self.click_radius = int(click_radius)
        self.with_flip = bool(with_flip)

        self.anchored = False
        self._image_np = None
        self._initial_prev_mask = None
        self._prev_prediction = None
        self._clicks: List[Tuple[int, int, bool]] = []

    def unanchor(self):
        self.anchored = False
        self._image_np = None
        self._initial_prev_mask = None
        self._prev_prediction = None
        self._clicks = []

    def _build_coord_features(self, h: int, w: int) -> np.ndarray:
        pos = np.zeros((h, w), dtype=np.float32)
        neg = np.zeros((h, w), dtype=np.float32)
        for x, y, is_pos in self._clicks[-self.max_clicks :]:
            if is_pos:
                cv2.circle(pos, (x, y), self.click_radius, 1.0, thickness=-1)
            else:
                cv2.circle(neg, (x, y), self.click_radius, 1.0, thickness=-1)
        return np.stack([pos, neg], axis=0)[None, ...]

    def _infer_logits(self, image_np: np.ndarray, prev_np: np.ndarray, coord_np: np.ndarray) -> np.ndarray:
        if self.with_prev_mask:
            image_in = np.concatenate([image_np, prev_np], axis=1)
        else:
            image_in = image_np
        return self.session.run(
            None,
            {
                self.image_input: image_in.astype(np.float32),
                self.coord_input: coord_np.astype(np.float32),
            },
        )[0].astype(np.float32)

    def _run(self) -> np.ndarray:
        prev = self._prev_prediction
        if prev is None:
            prev = self._initial_prev_mask
        if prev is None:
            prev = np.zeros((1, 1, self._image_np.shape[-2], self._image_np.shape[-1]), dtype=np.float32)

        coord = self._build_coord_features(self._image_np.shape[-2], self._image_np.shape[-1])
        logits = self._infer_logits(self._image_np, prev, coord)

        if self.with_flip:
            logits_flip = self._infer_logits(
                np.flip(self._image_np, axis=-1).copy(),
                np.flip(prev, axis=-1).copy(),
                np.flip(coord, axis=-1).copy(),
            )
            logits = 0.5 * (logits + np.flip(logits_flip, axis=-1).copy())

        self._prev_prediction = 1.0 / (1.0 + np.exp(-logits))
        return self._prev_prediction[:, 0]

    def interact(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        is_positive: bool,
        prev_mask: np.ndarray | None,
    ) -> np.ndarray:
        if not self.anchored:
            self._image_np = image.astype(np.float32, copy=True)
            self._initial_prev_mask = None if prev_mask is None else prev_mask.astype(np.float32, copy=True)
            self._prev_prediction = None
            self._clicks = []
            self.anchored = True

        self._clicks.append((int(x), int(y), bool(is_positive)))
        return self._run()

    def undo(self):
        if len(self._clicks) == 0:
            return None
        self._clicks.pop()
        if len(self._clicks) == 0:
            self._prev_prediction = None
            return None
        self._prev_prediction = None
        pred = self._run()
        return (pred > 0.5).astype(np.float32)
