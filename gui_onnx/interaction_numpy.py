from __future__ import annotations

from typing import Tuple

import numpy as np

from .click_controller_numpy import ClickControllerOnnxNumpy
from .interactive_utils_numpy import aggregate_wbg


class ClickInteractionOnnx:
    def __init__(
        self,
        image: np.ndarray,
        prev_mask: np.ndarray,
        true_size: Tuple[int, int],
        controller: ClickControllerOnnxNumpy,
        tar_obj: int,
    ):
        self.image = image
        self.prev_mask = prev_mask
        self.controller = controller
        self.h, self.w = true_size
        self.tar_obj = tar_obj
        self.first_click = True
        self.obj_mask = None
        self.out_prob = self.prev_mask.copy()

    def push_point(self, x: int, y: int, is_neg: bool) -> None:
        if self.first_click:
            last_obj_mask = self.prev_mask[self.tar_obj : self.tar_obj + 1][None, ...]
            self.obj_mask = self.controller.interact(
                self.image[None, ...],
                x,
                y,
                not is_neg,
                prev_mask=last_obj_mask,
            )
            self.first_click = False
        else:
            self.obj_mask = self.controller.interact(
                self.image[None, ...],
                x,
                y,
                not is_neg,
                prev_mask=None,
            )

    def predict(self) -> np.ndarray:
        self.out_prob = self.prev_mask.copy()
        self.out_prob = np.clip(self.out_prob, a_min=None, a_max=0.9)
        self.out_prob[self.tar_obj] = self.obj_mask[0]
        self.out_prob = aggregate_wbg(self.out_prob[1:], keep_bg=True, hard=True)
        return self.out_prob.astype(np.float32)
