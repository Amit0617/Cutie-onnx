from __future__ import annotations

import logging
import os
from os import path
from typing import Literal

import cv2
import numpy as np
from omegaconf import DictConfig, open_dict

from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.gui import GUI
from gui.resource_manager import ResourceManager

from .click_controller_numpy import ClickControllerOnnxNumpy
from .interaction_numpy import ClickInteractionOnnx
from .interactive_utils_numpy import (
    get_visualization,
    get_visualization_prob,
    image_to_chw_float,
    index_mask_to_one_hot,
    prob_to_mask,
)
from .onnx_inference_core_numpy import OnnxInferenceCoreNumpy
from .reader_numpy import PropagationReaderNumpy
from .sam2_click_controller_numpy import Sam2ClickControllerOnnxNumpy

log = logging.getLogger()


class MainControllerOnnxNumpy:
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.initialized = False

        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError("Either images, video, or workspace has to be specified")
            cfg["workspace"] = path.join(cfg["workspace_root"], basename)

        self.cfg = cfg
        self.num_objects = int(cfg["num_objects"])
        self.device = cfg["device"]
        self.amp = False

        self.initialize_networks()

        self.res_man = ResourceManager(cfg)
        if "workspace_init_only" in cfg and cfg["workspace_init_only"]:
            return
        self.processor = self.build_processor()
        self.gui = GUI(self, self.cfg)

        self.length = self.res_man.length
        self.interaction: ClickInteractionOnnx | None = None
        self.interaction_type = "Click"
        self.curr_ti = 0
        self.curr_object = 1
        self.propagating = False
        self.propagate_direction: Literal["forward", "backward", "none"] = "none"
        self.last_ex = 0
        self.last_ey = 0

        self.curr_frame_dirty = False
        self.curr_image_np = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_chw: np.ndarray | None = None
        self.curr_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob = np.zeros((self.num_objects + 1, self.h, self.w), dtype=np.float32)
        self.curr_prob[0] = 1.0

        self.vis_mode = "davis"
        self.vis_image = None
        self.save_visualization_mode = "None"
        self.save_soft_mask = False

        self.interacted_prob: np.ndarray | None = None
        self.overlay_layer: np.ndarray | None = None
        self.vis_target_objects = list(range(1, self.num_objects + 1))
        self.temp_box_prompt: tuple[int, int, int, int] | None = None

        self.load_current_image_mask()
        self.show_current_frame()

        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        self.output_fps = cfg["output_fps"]
        self.output_bitrate = cfg["output_bitrate"]

        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn
        self.gui.box_prompt_start_fn = self.on_box_prompt_start
        self.gui.box_prompt_update_fn = self.on_box_prompt_update
        self.gui.box_prompt_end_fn = self.on_box_prompt_end

        self.gui.show()
        self.gui.text("Initialized.")
        self.initialized = True

        self._try_load_layer("./docs/uiuc.png")
        self.gui.set_object_color(self.curr_object)
        self.update_config()

    def initialize_networks(self) -> None:
        backend = str(getattr(self.cfg, "click_backend_model", "ritm")).lower()
        if backend == "sam2":
            self.click_ctrl = Sam2ClickControllerOnnxNumpy(
                self.cfg.sam2_encoder_onnx,
                self.cfg.sam2_decoder_onnx,
                device=self.device,
            )
            log.info("Using SAM2 ONNX click backend for initial object selection.")
            return

        self.click_ctrl = ClickControllerOnnxNumpy(
            self.cfg.ritm_onnx,
            device=self.device,
            max_clicks=self.cfg.ritm_max_clicks,
            click_radius=self.cfg.ritm_click_radius,
            with_flip=True,
        )
        log.info("Using RITM ONNX click backend for initial object selection.")

    def build_processor(self):
        return OnnxInferenceCoreNumpy(self.cfg)

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        self.click_ctrl.unanchor()
        self.gui.text(f"Current object changed to {number}.")
        self.gui.set_object_color(number)
        self.show_current_frame()

    def click_fn(self, action: Literal["left", "right", "middle"], x: int, y: int):
        if self.propagating:
            return

        last_interaction = self.interaction
        if action in ["left", "right"]:
            self.convert_current_image_mask_numpy()
            image = self.curr_image_chw
            if last_interaction is None or last_interaction.tar_obj != self.curr_object:
                self.complete_interaction()
                self.click_ctrl.unanchor()
                self.interaction = ClickInteractionOnnx(
                    image,
                    self.curr_prob,
                    (self.h, self.w),
                    self.click_ctrl,
                    self.curr_object,
                )

            self.interaction.push_point(x, y, is_neg=(action == "right"))
            self.interacted_prob = self.interaction.predict()
            self.update_interacted_mask()
            self.update_gpu_gauges()
            return

        if action == "middle":
            target_object = int(self.curr_mask[int(y), int(x)])
            if target_object in self.vis_target_objects:
                self.vis_target_objects.remove(target_object)
            else:
                self.vis_target_objects.append(target_object)
            self.gui.text(f"Overlay target(s) changed to {self.vis_target_objects}")
            self.show_current_frame()
            return

        raise NotImplementedError(action)

    def _sam2_box_prompt_enabled(self) -> bool:
        return isinstance(self.click_ctrl, Sam2ClickControllerOnnxNumpy)

    def _normalized_box_prompt(
        self, x0: int | float, y0: int | float, x1: int | float, y1: int | float
    ) -> tuple[int, int, int, int]:
        x_min, x_max = sorted((int(round(x0)), int(round(x1))))
        y_min, y_max = sorted((int(round(y0)), int(round(y1))))
        return x_min, y_min, x_max, y_max

    def _draw_temp_box_prompt(self, image: np.ndarray) -> np.ndarray:
        if self.temp_box_prompt is None:
            return image
        x0, y0, x1, y1 = self._normalized_box_prompt(*self.temp_box_prompt)
        preview = image.copy()
        cv2.rectangle(preview, (x0, y0), (x1, y1), (80, 255, 80), 2)
        return preview

    def _ensure_box_interaction(self) -> ClickInteractionOnnx:
        last_interaction = self.interaction
        self.convert_current_image_mask_numpy()
        image = self.curr_image_chw
        if last_interaction is None or last_interaction.tar_obj != self.curr_object:
            self.complete_interaction()
            self.click_ctrl.unanchor()
            self.interaction = ClickInteractionOnnx(
                image,
                self.curr_prob,
                (self.h, self.w),
                self.click_ctrl,
                self.curr_object,
            )
        return self.interaction

    def on_box_prompt_start(self, x: int, y: int):
        if self.propagating or not self._sam2_box_prompt_enabled():
            return
        self.temp_box_prompt = self._normalized_box_prompt(x, y, x, y)
        self._show_preview_frame()

    def on_box_prompt_update(self, x: int, y: int):
        if self.temp_box_prompt is None or not self._sam2_box_prompt_enabled():
            return
        x0, y0, _, _ = self.temp_box_prompt
        self.temp_box_prompt = self._normalized_box_prompt(x0, y0, x, y)
        self._show_preview_frame()

    def on_box_prompt_end(self, x: int, y: int):
        if not self._sam2_box_prompt_enabled():
            self.temp_box_prompt = None
            return
        if self.propagating or self.temp_box_prompt is None:
            self.temp_box_prompt = None
            return

        x0, y0, _, _ = self.temp_box_prompt
        x0, y0, x1, y1 = self._normalized_box_prompt(x0, y0, x, y)
        self.temp_box_prompt = None
        if x1 <= x0 or y1 <= y0:
            self.gui.text("Box prompt ignored: drag a larger rectangle.")
            self.show_current_frame()
            return

        interaction = self._ensure_box_interaction()
        interaction.set_box(x0, y0, x1, y1)
        self.interacted_prob = interaction.predict()
        self.update_interacted_mask()
        self.update_gpu_gauges()

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_chw = None
        self.temp_box_prompt = None
        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_numpy(self, no_mask: bool = False):
        if self.curr_image_chw is None:
            self.curr_image_chw = image_to_chw_float(self.curr_image_np)
        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_mask_to_one_hot(self.curr_mask, self.num_objects + 1)

    def compose_current_im(self):
        self.vis_image = get_visualization(
            self.vis_mode,
            self.curr_image_np,
            self.curr_mask,
            self.overlay_layer,
            self.vis_target_objects,
        )
        self.vis_image = self._draw_temp_box_prompt(self.vis_image)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        self.vis_image = get_visualization_prob(
            self.vis_mode,
            self.curr_image_np,
            self.curr_prob,
            self.overlay_layer,
            self.vis_target_objects,
        )
        self.vis_image = self._draw_temp_box_prompt(self.vis_image)
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            "Propagation only (higher quality)",
            "Always",
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob)
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == "Always":
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + ".jpg")

    def _show_preview_frame(self):
        self.compose_current_im()
        self.update_canvas()
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + ".jpg")

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def on_slider_update(self):
        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False
            self.reset_this_interaction()
            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def on_forward_propagation(self):
        if self.propagating:
            self.propagating = False
            self.propagate_direction = "none"
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = "forward"
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            self.propagating = False
            self.propagate_direction = "none"
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = "backward"
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f"Propagation stopped at t={self.curr_ti}.")
        self.gui.pause_propagation()

    def on_propagate(self):
        self.convert_current_image_mask_numpy()
        self.gui.text(f"Propagation started at t={self.curr_ti}.")
        self.processor.clear_sensory_memory()
        self.curr_prob = self.processor.step(self.curr_image_chw, self.curr_prob[1:], idx_mask=False)
        self.curr_mask = prob_to_mask(self.curr_prob)
        self.interacted_prob = None
        self.reset_this_interaction()
        self.show_current_frame(fast=True, invalid_soft_mask=True)

        self.propagating = True
        self.gui.clear_all_mem_button.setEnabled(False)
        self.gui.clear_non_perm_mem_button.setEnabled(False)
        self.gui.tl_slider.setEnabled(False)

        dataset = PropagationReaderNumpy(self.res_man, self.curr_ti, self.propagate_direction)
        for image_np, image_chw in dataset:
            if not self.propagating:
                break
            self.curr_image_np = image_np
            self.curr_image_chw = image_chw
            self.propagate_fn()
            self.curr_prob = self.processor.step(self.curr_image_chw)
            self.curr_mask = prob_to_mask(self.curr_prob)

            self.save_current_mask()
            self.show_current_frame(fast=True)

            self.update_memory_gauges()
            self.gui.process_events()

            if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                break

        self.propagating = False
        self.curr_frame_dirty = False
        self.on_pause()
        self.on_slider_update()
        self.gui.process_events()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            self.load_current_image_mask()
        else:
            self.complete_interaction()
            self.update_interacted_mask()

        self.convert_current_image_mask_numpy()
        self.gui.text(f"Permanent memory saved at {self.curr_ti}.")
        self.curr_prob = self.processor.step(
            self.curr_image_chw,
            self.curr_prob[1:],
            idx_mask=False,
            force_permanent=True,
        )
        self.update_memory_gauges()
        self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        image_folder = path.join(self.cfg["workspace"], "visualization", self.vis_mode)
        save_folder = self.cfg["workspace"]
        if path.exists(image_folder):
            output_path = path.join(save_folder, f"visualization_{self.vis_mode}.mp4")
            self.gui.text("Exporting visualization -- please wait")
            self.gui.process_events()
            convert_frames_to_video(
                image_folder,
                output_path,
                fps=self.output_fps,
                bitrate=self.output_bitrate,
                progress_callback=self.gui.progressbar_update,
            )
            self.gui.text(f"Visualization exported to {output_path}")
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f"No visualization images found in {image_folder}")

    def on_export_binary(self):
        mask_folder = path.join(self.cfg["workspace"], "masks")
        save_folder = path.join(self.cfg["workspace"], "binary_masks")
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text("Exporting binary masks -- please wait")
            self.gui.process_events()
            convert_mask_to_binary(
                mask_folder,
                save_folder,
                self.vis_target_objects,
                progress_callback=self.gui.progressbar_update,
            )
            self.gui.text(f"Binary masks exported to {save_folder}")
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f"No masks found in {mask_folder}")

    def on_object_dial_change(self):
        self.hit_number_key(self.gui.object_dial.value())

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = prob_to_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        self.temp_box_prompt = None
        self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill(0)
        self.curr_frame_dirty = True
        self.save_current_mask()



        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        self.gui.tl_slider.setValue(max(0, self.curr_ti - step))

    def on_next_frame(self, step=1):
        self.gui.tl_slider.setValue(min(self.curr_ti + step, self.length - 1))

    def update_gpu_gauges(self):
        label = "ONNX Runtime"
        if self.device == "cuda":
            label = "ONNX Runtime CUDA"
        self.gui.gpu_mem_gauge.setFormat(label)
        self.gui.gpu_mem_gauge.setValue(0)
        self.gui.torch_mem_gauge.setFormat("N/A")
        self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f"{curr_perm_tokens} / {curr_perm_tokens}")
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = max(1, int(self.processor.memory.max_work_tokens))
            max_long_tokens = max(1, int(self.processor.memory.max_long_tokens))
            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f"{curr_work_tokens} / {max_work_tokens}")
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))
            self.gui.long_mem_gauge.setFormat(f"{curr_long_tokens} / {max_long_tokens}")
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))
        except AttributeError:
            self.gui.work_mem_gauge.setFormat("Unknown")
            self.gui.long_mem_gauge.setFormat("Unknown")
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(), self.gui.work_mem_max.value() - 1)
            )
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(), self.gui.work_mem_min.value() + 1)
            )
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term["min_mem_frames"] = self.gui.work_mem_min.value()
                self.cfg.long_term["max_mem_frames"] = self.gui.work_mem_max.value()
                self.cfg.long_term["max_num_tokens"] = self.gui.long_mem_max.value()
                self.cfg["mem_every"] = self.gui.mem_every_box.value()
            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file("Mask")
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))
        shape_condition = len(mask.shape) == 2 and mask.shape[-1] == self.w and mask.shape[-2] == self.h
        object_condition = mask.max() <= self.num_objects

        if not shape_condition:
            self.gui.text(f"Expected ({self.h}, {self.w}). Got {mask.shape} instead.")
        elif not object_condition:
            self.gui.text(
                f"Expected {self.num_objects} objects. Got {mask.max()} objects instead."
            )
        else:
            self.gui.text(f"Mask file {file_name} loaded.")
            self.curr_image_chw = None
            self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self.gui.open_file("Layer")
        if len(file_name) == 0:
            return
        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))
            self.gui.text(f"Layer file {file_name} loaded.")
            self.overlay_layer = layer
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f"{file_name} not found.")

    def on_set_save_visualization_mode(self):
        self.save_visualization_mode = self.gui.save_visualization_combo.currentText()

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T


__all__ = ["MainControllerOnnxNumpy"]
