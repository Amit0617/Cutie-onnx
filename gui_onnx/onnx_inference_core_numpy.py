from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np


class _MemStoreCounter:
    def __init__(self) -> None:
        self._perm = 0
        self._non_perm = 0

    def set_tokens(self, perm: int, non_perm: int) -> None:
        self._perm = int(max(0, perm))
        self._non_perm = int(max(0, non_perm))

    def perm_size(self, bucket_id: int) -> int:
        return self._perm

    def non_perm_size(self, bucket_id: int) -> int:
        return self._non_perm


class _MemoryGaugeProxy:
    def __init__(self, cfg) -> None:
        self.min_mem_frames = int(cfg.long_term.min_mem_frames - 1)
        self.max_mem_frames = int(cfg.long_term.max_mem_frames - 1)
        self.max_long_tokens = int(cfg.long_term.max_num_tokens)
        self.max_work_tokens = 1
        self.work_mem = _MemStoreCounter()
        self.long_mem = _MemStoreCounter()

    def update(self, hw: int, total_frames: int, perm_frames: int) -> None:
        self.max_work_tokens = int(max(1, self.max_mem_frames * hw))
        perm = int(max(0, perm_frames) * hw)
        non_perm = int(max(0, total_frames - perm_frames) * hw)
        self.work_mem.set_tokens(perm, non_perm)
        self.long_mem.set_tokens(0, 0)


@dataclass
class _FrameEntry:
    key: np.ndarray
    shrinkage: np.ndarray
    mask_value: np.ndarray
    object_memory: np.ndarray
    permanent: bool


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-7, None)


def _aggregate(prob: np.ndarray, axis: int) -> np.ndarray:
    new_prob = np.concatenate(
        [np.prod(1.0 - prob, axis=axis, keepdims=True), prob],
        axis=axis,
    ).clip(1e-7, 1.0 - 1e-7)
    return np.log(new_prob / (1.0 - new_prob))


def _pad_divide_by_chw(image: np.ndarray, divisor: int):
    h, w = image.shape[-2:]
    new_h = h if h % divisor == 0 else h + divisor - h % divisor
    new_w = w if w % divisor == 0 else w + divisor - w % divisor
    lh = int((new_h - h) / 2)
    uh = int(new_h - h) - lh
    lw = int((new_w - w) / 2)
    uw = int(new_w - w) - lw
    pad = (lw, uw, lh, uh)
    padded = np.pad(image, ((0, 0), (lh, uh), (lw, uw)), mode="constant")
    return padded, pad


def _pad_divide_by_hw(mask: np.ndarray, divisor: int):
    h, w = mask.shape[-2:]
    new_h = h if h % divisor == 0 else h + divisor - h % divisor
    new_w = w if w % divisor == 0 else w + divisor - w % divisor
    lh = int((new_h - h) / 2)
    uh = int(new_h - h) - lh
    lw = int((new_w - w) / 2)
    uw = int(new_w - w) - lw
    pad = (lw, uw, lh, uh)
    padded = np.pad(mask, ((lh, uh), (lw, uw)), mode="constant")
    return padded, pad


def _unpad_chw(image: np.ndarray, pad):
    lw, uw, lh, uh = pad
    h_slice = slice(lh, None if uh == 0 else -uh)
    w_slice = slice(lw, None if uw == 0 else -uw)
    return image[:, h_slice, w_slice]


def _resize_chw(image: np.ndarray, size, interpolation) -> np.ndarray:
    h, w = size
    hwc = image.transpose(1, 2, 0)
    resized = cv2.resize(hwc, (w, h), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized.transpose(2, 0, 1).astype(np.float32)


class OnnxInferenceCoreNumpy:
    def __init__(self, cfg) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModuleNotFoundError("onnxruntime is required for the ONNX backend.") from exc

        self.cfg = cfg
        self.device = cfg.device
        self.mem_every = int(cfg.mem_every)
        self.max_internal_size = int(cfg.max_internal_size)
        self.num_objects_gui = int(cfg.num_objects)

        encoder_path = Path(getattr(cfg, "onnx_encoder", "weights/cutie_image_encoder.onnx"))
        write_path = Path(getattr(cfg, "onnx_memory_write", "weights/cutie_memory_write.onnx"))
        read_path = Path(getattr(cfg, "onnx_read_decode", "weights/cutie_read_decode.onnx"))
        for model_path in (encoder_path, write_path, read_path):
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(str(encoder_path), providers=providers)
        self.memory_write = ort.InferenceSession(str(write_path), providers=providers)
        self.read_decode = ort.InferenceSession(str(read_path), providers=providers)

        image_shape = self.memory_write.get_inputs()[0].shape
        sensory_shape = self.memory_write.get_inputs()[2].shape
        self.input_h = int(image_shape[2])
        self.input_w = int(image_shape[3])
        self.num_objects_onnx = int(sensory_shape[1])
        if self.num_objects_gui > self.num_objects_onnx:
            raise ValueError(
                f"GUI requested {self.num_objects_gui} objects, but ONNX supports {self.num_objects_onnx}."
            )

        rd_inputs = {x.name: x for x in self.read_decode.get_inputs()}
        self.h16 = int(rd_inputs["key"].shape[2])
        self.w16 = int(rd_inputs["key"].shape[3])
        self.memory_frames = int(rd_inputs["memory_key"].shape[2])
        self.sensory_channels = int(self.memory_write.get_inputs()[2].shape[2])

        self.memory = _MemoryGaugeProxy(cfg)
        self.curr_ti = -1
        self.last_mem_ti = 0
        self._frames: List[_FrameEntry] = []
        self._sensory: Optional[np.ndarray] = None
        self._last_mask = np.zeros((1, self.num_objects_onnx, self.h16, self.w16), dtype=np.float32)
        self._selector = np.zeros((1, self.num_objects_onnx, 1, 1), dtype=np.float32)
        self._selector[:, : self.num_objects_gui] = 1.0

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self._frames = []
        self._sensory = None
        self._last_mask.fill(0)
        self.memory.update(self.h16 * self.w16, 0, 0)

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self._frames = [item for item in self._frames if item.permanent]
        self.memory.update(
            self.h16 * self.w16,
            len(self._frames),
            sum(1 for item in self._frames if item.permanent),
        )

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self._sensory = None
        self._last_mask.fill(0)

    def update_config(self, cfg):
        self.mem_every = int(cfg["mem_every"])
        self.memory.min_mem_frames = int(cfg.long_term.min_mem_frames - 1)
        self.memory.max_mem_frames = int(cfg.long_term.max_mem_frames - 1)
        self.memory.max_long_tokens = int(cfg.long_term.max_num_tokens)
        self.memory.update(
            self.h16 * self.w16,
            len(self._frames),
            sum(1 for item in self._frames if item.permanent),
        )

    def _transform_image(self, image: np.ndarray):
        original_h, original_w = image.shape[-2:]
        resized = image
        resized_h, resized_w = original_h, original_w
        resize_needed = False
        if self.max_internal_size > 0:
            min_side = min(original_h, original_w)
            if min_side > self.max_internal_size:
                resize_needed = True
                resized_h = int(original_h / min_side * self.max_internal_size)
                resized_w = int(original_w / min_side * self.max_internal_size)
                resized = _resize_chw(resized, (resized_h, resized_w), cv2.INTER_LINEAR)

        padded, pad = _pad_divide_by_chw(resized, 16)
        src_h, src_w = padded.shape[-2:]
        scale = min(self.input_h / src_h, self.input_w / src_w)
        fit_h = min(self.input_h, max(1, int(round(src_h * scale))))
        fit_w = min(self.input_w, max(1, int(round(src_w * scale))))
        fitted = _resize_chw(padded, (fit_h, fit_w), cv2.INTER_LINEAR)
        canvas = np.zeros((3, self.input_h, self.input_w), dtype=np.float32)
        top = (self.input_h - fit_h) // 2
        left = (self.input_w - fit_w) // 2
        canvas[:, top : top + fit_h, left : left + fit_w] = fitted
        meta = {
            "original_h": original_h,
            "original_w": original_w,
            "resize_needed": resize_needed,
            "resized_h": resized_h,
            "resized_w": resized_w,
            "pad": pad,
            "padded_h": src_h,
            "padded_w": src_w,
            "canvas_top": top,
            "canvas_left": left,
            "canvas_h": fit_h,
            "canvas_w": fit_w,
        }
        return canvas, meta

    def _transform_mask(self, mask: np.ndarray, *, idx_mask: bool, meta):
        if idx_mask:
            out = mask
            if meta["resize_needed"]:
                out = cv2.resize(
                    out.astype(np.float32),
                    (meta["resized_w"], meta["resized_h"]),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                ).astype(np.int64)
            out, _ = _pad_divide_by_hw(out, 16)
            fitted = cv2.resize(
                out.astype(np.float32),
                (meta["canvas_w"], meta["canvas_h"]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ).astype(np.int64)
            canvas = np.zeros((self.input_h, self.input_w), dtype=np.int64)
            top = meta["canvas_top"]
            left = meta["canvas_left"]
            canvas[top : top + meta["canvas_h"], left : left + meta["canvas_w"]] = fitted
            return canvas

        out = mask.astype(np.float32)
        if meta["resize_needed"]:
            out = _resize_chw(out, (meta["resized_h"], meta["resized_w"]), cv2.INTER_LINEAR)
        padded_channels = []
        for channel in out:
            padded_channel, _ = _pad_divide_by_hw(channel, 16)
            padded_channels.append(padded_channel)
        out = np.stack(padded_channels, axis=0)
        fitted = _resize_chw(out, (meta["canvas_h"], meta["canvas_w"]), cv2.INTER_LINEAR)
        canvas = np.zeros((out.shape[0], self.input_h, self.input_w), dtype=np.float32)
        top = meta["canvas_top"]
        left = meta["canvas_left"]
        canvas[:, top : top + meta["canvas_h"], left : left + meta["canvas_w"]] = fitted
        return canvas

    def _restore_output(self, prob: np.ndarray, meta) -> np.ndarray:
        top = meta["canvas_top"]
        left = meta["canvas_left"]
        prob = prob[:, top : top + meta["canvas_h"], left : left + meta["canvas_w"]]
        if prob.shape[-2:] != (meta["padded_h"], meta["padded_w"]):
            prob = _resize_chw(prob, (meta["padded_h"], meta["padded_w"]), cv2.INTER_LINEAR)
        prob = _unpad_chw(prob, meta["pad"])
        if meta["resize_needed"]:
            prob = _resize_chw(prob, (meta["original_h"], meta["original_w"]), cv2.INTER_LINEAR)
        return prob

    def _pad_to_onnx_objects(self, mask: np.ndarray) -> np.ndarray:
        out = np.zeros((self.num_objects_onnx, mask.shape[-2], mask.shape[-1]), dtype=np.float32)
        out[: self.num_objects_gui] = mask[: self.num_objects_gui].astype(np.float32)
        return out

    def _onnx_to_gui_channels(self, prob: np.ndarray) -> np.ndarray:
        if self.num_objects_gui == self.num_objects_onnx:
            return prob
        fg = prob[1 : self.num_objects_gui + 1]
        bg = prob[:1] + np.sum(prob[self.num_objects_gui + 1 :], axis=0, keepdims=True)
        out = np.concatenate([bg, fg], axis=0)
        return out / np.clip(np.sum(out, axis=0, keepdims=True), 1e-7, None)

    def _build_memory_inputs(self):
        pad_len = max(0, self.memory_frames - len(self._frames))
        key_dim = self._frames[-1].key.shape[0]
        value_dim = self._frames[-1].mask_value.shape[1]
        q_dim = self._frames[-1].object_memory.shape[1]
        e_dim = self._frames[-1].object_memory.shape[2]

        mem_key = np.zeros((1, key_dim, self.memory_frames, self.h16, self.w16), dtype=np.float32)
        mem_shrink = np.zeros((1, 1, self.memory_frames, self.h16, self.w16), dtype=np.float32)
        mem_val = np.zeros(
            (1, self.num_objects_onnx, value_dim, self.memory_frames, self.h16, self.w16),
            dtype=np.float32,
        )
        obj_mem = np.zeros(
            (1, self.num_objects_onnx, self.memory_frames, q_dim, e_dim),
            dtype=np.float32,
        )
        for index, frame in enumerate(self._frames[-self.memory_frames :]):
            t = index + pad_len
            mem_key[:, :, t] = frame.key
            mem_shrink[:, :, t] = frame.shrinkage
            mem_val[:, :, :, t] = frame.mask_value
            obj_mem[:, :, t] = frame.object_memory
        return mem_key, mem_shrink, mem_val, obj_mem

    def _append_memory(self, key, shrinkage, mask_value, object_memory, *, permanent: bool):
        self._frames.append(
            _FrameEntry(
                key=key[0],
                shrinkage=shrinkage[0],
                mask_value=mask_value[0],
                object_memory=object_memory[0],
                permanent=permanent,
            )
        )
        while len(self._frames) > self.memory_frames:
            non_perm = [index for index, frame in enumerate(self._frames) if not frame.permanent]
            drop_index = non_perm[0] if len(non_perm) > 0 else 0
            self._frames.pop(drop_index)
        self.memory.update(
            self.h16 * self.w16,
            len(self._frames),
            sum(1 for frame in self._frames if frame.permanent),
        )

    def step(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        objects: Optional[Sequence[int]] = None,
        *,
        idx_mask: bool = True,
        end: bool = False,
        delete_buffer: bool = True,
        force_permanent: bool = False,
    ) -> np.ndarray:
        del objects, delete_buffer
        self.curr_ti += 1
        image_onnx, meta = self._transform_image(image)
        image_np = image_onnx[None, ...].astype(np.float32)

        enc_out = self.encoder.run(None, {"image": image_np})
        ms_feat_16x, ms_feat_8x, ms_feat_4x, pix_feat, key, shrinkage, selection = enc_out

        is_mem_frame = (((self.curr_ti - self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end))

        if mask is not None:
            mask = self._transform_mask(mask, idx_mask=idx_mask, meta=meta)
            if idx_mask:
                one_hot = [(mask == obj_id).astype(np.float32) for obj_id in range(1, self.num_objects_gui + 1)]
                mask_no_bg = np.stack(one_hot, axis=0) if len(one_hot) > 0 else np.empty((0, *mask.shape), dtype=np.float32)
            else:
                mask_no_bg = mask.astype(np.float32)
            mask_no_bg = self._pad_to_onnx_objects(mask_no_bg)
            logits = _aggregate(mask_no_bg, axis=0)
            pred_prob_with_bg = _softmax(logits, axis=0).astype(np.float32)
        else:
            if len(self._frames) == 0:
                out = np.zeros((self.num_objects_gui + 1, meta["original_h"], meta["original_w"]), dtype=np.float32)
                out[0] = 1.0
                return out

            mem_key, mem_shrink, mem_val, obj_mem = self._build_memory_inputs()
            if self._sensory is None:
                self._sensory = np.zeros(
                    (1, self.num_objects_onnx, mem_val.shape[2], self.h16, self.w16),
                    dtype=np.float32,
                )
            rd_out = self.read_decode.run(
                None,
                {
                    "ms_feat_16x": ms_feat_16x,
                    "ms_feat_8x": ms_feat_8x,
                    "ms_feat_4x": ms_feat_4x,
                    "pix_feat": pix_feat,
                    "key": key,
                    "selection": selection,
                    "memory_key": mem_key,
                    "memory_shrinkage": mem_shrink,
                    "memory_mask_value": mem_val,
                    "object_memory": obj_mem,
                    "sensory": self._sensory,
                    "last_mask": self._last_mask,
                    "selector": self._selector,
                },
            )
            self._sensory = rd_out[1]
            pred_prob_with_bg = rd_out[3][0].astype(np.float32)

        if is_mem_frame or force_permanent:
            if self._sensory is None:
                self._sensory = np.zeros(
                    (1, self.num_objects_onnx, self.sensory_channels, self.h16, self.w16),
                    dtype=np.float32,
                )
            masks_for_memory = pred_prob_with_bg[1 : self.num_objects_onnx + 1][None, ...].astype(np.float32)
            msk_value, new_sensory, obj_memory = self.memory_write.run(
                None,
                {
                    "image": image_np,
                    "pix_feat": pix_feat,
                    "sensory": self._sensory,
                    "masks": masks_for_memory,
                },
            )
            self._sensory = new_sensory
            permanent = bool(force_permanent) or len(self._frames) == 0
            self._append_memory(key, shrinkage, msk_value, obj_memory, permanent=permanent)
            self.last_mem_ti = self.curr_ti

        last_mask = pred_prob_with_bg[1 : self.num_objects_onnx + 1]
        last_mask = _resize_chw(last_mask, (self.h16, self.w16), cv2.INTER_LINEAR)
        self._last_mask = last_mask[None, ...].astype(np.float32)

        pred_prob_with_bg = self._onnx_to_gui_channels(pred_prob_with_bg)
        output_prob = self._restore_output(pred_prob_with_bg, meta)
        output_prob = output_prob / np.clip(np.sum(output_prob, axis=0, keepdims=True), 1e-7, None)
        return output_prob.astype(np.float32)
