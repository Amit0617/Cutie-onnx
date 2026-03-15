from __future__ import annotations

from typing import List, Literal

import numpy as np

from cutie.utils.palette import davis_palette


def image_to_chw_float(image: np.ndarray) -> np.ndarray:
    return image.transpose(2, 0, 1).astype(np.float32) / 255.0


def prob_to_mask(prob: np.ndarray) -> np.ndarray:
    return np.argmax(prob, axis=0).astype(np.uint8)


def index_mask_to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[mask].transpose(2, 0, 1)


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-7, None)


def aggregate_wbg(prob: np.ndarray, keep_bg: bool = False, hard: bool = False) -> np.ndarray:
    new_prob = np.concatenate(
        [np.prod(1.0 - prob, axis=0, keepdims=True), prob],
        axis=0,
    ).clip(1e-7, 1.0 - 1e-7)
    logits = np.log(new_prob / (1.0 - new_prob))
    if hard:
        logits *= 1000.0
    out = _softmax(logits, axis=0)
    return out if keep_bg else out[1:]


color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
color_map_np = (color_map_np.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
grayscale_weights = np.array([[0.3, 0.59, 0.11]], dtype=np.float32)


def _target_prob(prob: np.ndarray, target_objects: List[int]) -> np.ndarray:
    if len(target_objects) == 0:
        return np.zeros(prob.shape[1:] + (1,), dtype=np.float32)
    return np.sum(prob[np.asarray(target_objects, dtype=np.int32)], axis=0, keepdims=False)[
        ..., None
    ]


def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, fade: bool = False):
    im_overlay = image.copy()
    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1.0 - alpha) * colored_mask
    binary_mask = mask > 0
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def overlay_popup(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    im_overlay = image.copy()
    binary_mask = ~np.isin(mask, target_objects)
    colored_region = (im_overlay[binary_mask] * grayscale_weights).sum(-1, keepdims=True)
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)


def overlay_layer(image: np.ndarray, mask: np.ndarray, layer: np.ndarray, target_objects: List[int]):
    obj_mask = np.isin(mask, target_objects).astype(np.float32)[:, :, None]
    layer_alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
    layer_rgb = layer[:, :, :3].astype(np.float32)
    background_alpha = (1.0 - obj_mask) * (1.0 - layer_alpha)
    out = image.astype(np.float32) * background_alpha
    out += layer_rgb * (1.0 - obj_mask) * layer_alpha
    out += image.astype(np.float32) * obj_mask
    return out.clip(0, 255).astype(image.dtype)


def overlay_rgba(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    obj_mask = np.isin(mask, target_objects).astype(np.float32)[:, :, None] * 255.0
    return np.concatenate([image, obj_mask.astype(np.uint8)], axis=-1)


def overlay_popup_prob(image: np.ndarray, prob: np.ndarray, target_objects: List[int]):
    image_f = image.astype(np.float32) / 255.0
    obj_mask = _target_prob(prob, target_objects)
    gray = (image_f * grayscale_weights).sum(-1, keepdims=True)
    out = obj_mask * image_f + (1.0 - obj_mask) * gray
    return (out.clip(0, 1) * 255.0).astype(np.uint8)


def overlay_layer_prob(image: np.ndarray, prob: np.ndarray, layer: np.ndarray, target_objects: List[int]):
    image_f = image.astype(np.float32) / 255.0
    obj_mask = _target_prob(prob, target_objects)
    layer_alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
    layer_rgb = layer[:, :, :3].astype(np.float32) / 255.0
    background_alpha = (1.0 - obj_mask) * (1.0 - layer_alpha)
    out = image_f * background_alpha + layer_rgb * (1.0 - obj_mask) * layer_alpha + image_f * obj_mask
    return (out.clip(0, 1) * 255.0).astype(np.uint8)


def overlay_rgba_prob(image: np.ndarray, prob: np.ndarray, target_objects: List[int]):
    obj_mask = (_target_prob(prob, target_objects) * 255.0).clip(0, 255).astype(np.uint8)
    return np.concatenate([image, obj_mask], axis=-1)


def get_visualization(
    mode: Literal["image", "mask", "fade", "davis", "light", "popup", "layer", "rgba"],
    image: np.ndarray,
    mask: np.ndarray,
    layer: np.ndarray,
    target_objects: List[int],
) -> np.ndarray:
    if mode == "image":
        return image
    if mode == "mask":
        return color_map_np[mask]
    if mode == "fade":
        return overlay_davis(image, mask, fade=True)
    if mode == "davis":
        return overlay_davis(image, mask)
    if mode == "light":
        return overlay_davis(image, mask, 0.9)
    if mode == "popup":
        return overlay_popup(image, mask, target_objects)
    if mode == "layer":
        if layer is None:
            return overlay_davis(image, mask)
        return overlay_layer(image, mask, layer, target_objects)
    if mode == "rgba":
        return overlay_rgba(image, mask, target_objects)
    raise NotImplementedError(mode)


def get_visualization_prob(
    mode: Literal["image", "mask", "fade", "davis", "light", "popup", "layer", "rgba"],
    image: np.ndarray,
    prob: np.ndarray,
    layer: np.ndarray,
    target_objects: List[int],
) -> np.ndarray:
    if mode in {"image", "mask", "fade", "davis", "light"}:
        return get_visualization(mode, image, prob_to_mask(prob), layer, target_objects)
    if mode == "popup":
        return overlay_popup_prob(image, prob, target_objects)
    if mode == "layer":
        if layer is None:
            return get_visualization("davis", image, prob_to_mask(prob), layer, target_objects)
        return overlay_layer_prob(image, prob, layer, target_objects)
    if mode == "rgba":
        return overlay_rgba_prob(image, prob, target_objects)
    raise NotImplementedError(mode)
