from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np


class Sam2ClickControllerOnnxNumpy:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        device: str = "cpu",
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for the ONNX SAM2 click backend."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)

        encoder_inputs = self.encoder_session.get_inputs()
        self.encoder_input_name = encoder_inputs[0].name
        self.encoder_input_height = int(encoder_inputs[0].shape[2])
        self.encoder_input_width = int(encoder_inputs[0].shape[3])
        self.encoder_output_names = [item.name for item in self.encoder_session.get_outputs()]

        decoder_inputs = self.decoder_session.get_inputs()
        self.decoder_input_names = [item.name for item in decoder_inputs]
        self.decoder_output_names = [item.name for item in self.decoder_session.get_outputs()]

        self.scale_factor = 4
        self.anchored = False
        self._clicks: List[Tuple[int, int, bool]] = []
        self._box_prompt: Tuple[int, int, int, int] | None = None
        self._image_size: Tuple[int, int] | None = None
        self._encoder_outputs: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def unanchor(self):
        self.anchored = False
        self._clicks = []
        self._box_prompt = None
        self._image_size = None
        self._encoder_outputs = None

    def _prepare_encoder_input(self, image_np: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_np, (self.encoder_input_width, self.encoder_input_height))
        resized = resized.astype(np.float32)
        if resized.max() > 1.0:
            resized = resized / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        return normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def _ensure_anchored(self, image: np.ndarray) -> None:
        if self.anchored:
            return

        image_np = np.asarray(image)
        if image_np.ndim != 4 or image_np.shape[0] != 1 or image_np.shape[1] != 3:
            raise ValueError("SAM2 click backend expects image input shaped as (1, 3, H, W).")

        # GUI uses RGB float image tensors in [0, 1].
        hwc_image = np.clip(image_np[0].transpose(1, 2, 0), 0.0, 1.0)
        self._image_size = (int(hwc_image.shape[0]), int(hwc_image.shape[1]))
        encoder_input = self._prepare_encoder_input(hwc_image)
        outputs = self.encoder_session.run(
            self.encoder_output_names,
            {self.encoder_input_name: encoder_input},
        )
        self._encoder_outputs = (outputs[0], outputs[1], outputs[2])
        self._clicks = []
        self._box_prompt = None
        self.anchored = True

    def _prepare_points(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._image_size is None:
            raise RuntimeError("SAM2 click backend is not anchored to an image.")

        coords_list: List[List[float]] = []
        labels_list: List[float] = []
        if self._box_prompt is not None:
            x0, y0, x1, y1 = self._box_prompt
            coords_list.extend([[x0, y0], [x1, y1]])
            labels_list.extend([2.0, 3.0])
        for x, y, is_pos in self._clicks:
            coords_list.append([x, y])
            labels_list.append(1.0 if is_pos else 0.0)

        coords = np.array(coords_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)

        coords = coords[None, ...]
        labels = labels[None, ...]
        coords[..., 0] = coords[..., 0] / self._image_size[1] * self.encoder_input_width
        coords[..., 1] = coords[..., 1] / self._image_size[0] * self.encoder_input_height
        return coords.astype(np.float32), labels.astype(np.float32)

    def _decode(self) -> np.ndarray:
        if self._encoder_outputs is None or self._image_size is None:
            raise RuntimeError("SAM2 click backend is missing cached encoder outputs.")

        high_res_feats_0, high_res_feats_1, image_embedding = self._encoder_outputs
        point_coords, point_labels = self._prepare_points()

        mask_input = np.zeros(
            (
                point_labels.shape[0],
                1,
                self.encoder_input_height // self.scale_factor,
                self.encoder_input_width // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        decoder_inputs = {
            self.decoder_input_names[0]: image_embedding,
            self.decoder_input_names[1]: high_res_feats_0,
            self.decoder_input_names[2]: high_res_feats_1,
            self.decoder_input_names[3]: point_coords,
            self.decoder_input_names[4]: point_labels,
            self.decoder_input_names[5]: mask_input,
            self.decoder_input_names[6]: has_mask_input,
        }
        masks, scores = self.decoder_session.run(self.decoder_output_names, decoder_inputs)[:2]

        best_mask = masks[0, int(np.argmax(scores[0]))]
        best_mask = cv2.resize(best_mask, (self._image_size[1], self._image_size[0]))
        prob = 1.0 / (1.0 + np.exp(-best_mask))
        return prob[None, ...].astype(np.float32)

    def interact(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        is_positive: bool,
        prev_mask: np.ndarray | None,
    ) -> np.ndarray:
        del prev_mask
        self._ensure_anchored(image)
        self._clicks.append((int(x), int(y), bool(is_positive)))
        point_type = "positive" if is_positive else "negative"
        print(f"SAM2 click: x={int(x)} y={int(y)} type={point_type}", flush=True)
        return self._decode()

    def set_box(
        self,
        image: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        prev_mask: np.ndarray | None,
    ) -> np.ndarray:
        del prev_mask
        self._ensure_anchored(image)
        x_min, x_max = sorted((int(x0), int(x1)))
        y_min, y_max = sorted((int(y0), int(y1)))
        self._box_prompt = (x_min, y_min, x_max, y_max)
        print(
            f"SAM2 box: x0={x_min} y0={y_min} x1={x_max} y1={y_max}",
            flush=True,
        )
        return self._decode()

    def undo(self):
        if len(self._clicks) == 0:
            return None
        self._clicks.pop()
        if len(self._clicks) == 0:
            return None
        pred = self._decode()
        return (pred > 0.5).astype(np.float32)
