from __future__ import annotations

from typing import Iterator, Literal, Tuple

import numpy as np

from gui.resource_manager import ResourceManager

from .interactive_utils_numpy import image_to_chw_float


class PropagationReaderNumpy:
    def __init__(self, res_man: ResourceManager, start_ti: int, direction: Literal["forward", "backward"]):
        self.res_man = res_man
        self.start_ti = start_ti
        self.direction = direction

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.direction == "forward":
            frame_range = range(self.start_ti + 1, self.res_man.T)
        elif self.direction == "backward":
            frame_range = range(self.start_ti - 1, -1, -1)
        else:
            raise NotImplementedError(self.direction)

        for ti in frame_range:
            image = self.res_man.get_image(ti)
            yield image, image_to_chw_float(image)
