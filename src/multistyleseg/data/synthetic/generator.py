from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import IterableDataset

from multistyleseg.data.synthetic.utils import (
    AnnotationType,
    color_mapping,
    synthesize_image,
    Task,
)

cv2.setNumThreads(0)


class SynthTriangle(IterableDataset):
    def __init__(
        self,
        resolution: int = 128,
        n_shapes: int = 10,
        annotation_types: Tuple[AnnotationType] = (
            AnnotationType.FINE,
            AnnotationType.COARSE,
        ),
        return_all_styles: bool = False,
        task: Task = Task.COLOR_BASED,
    ):
        super().__init__()

        self.resolution = resolution
        self.n_shapes = n_shapes
        if not isinstance(annotation_types, (list, tuple)):
            annotation_types = [annotation_types]
        self.annotation_types = annotation_types
        self.return_all_styles = return_all_styles
        self.task = task

    def __iter__(self):
        while True:
            annotation_type = np.random.choice(self.annotation_types)
            data = synthesize_image(
                self.resolution,
                annotation_type,
                self.n_shapes,
                self.task,
            )
            data["expected_style"] = annotation_type.value
            data["task"] = self.task.value
            yield data

    def __getitem__(self, index):
        return self.get_one()

    def get_one(self):
        return next(iter(self))
