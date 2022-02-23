import os.path as osp
import sys
from collections import Counter, defaultdict
from typing import Optional
import albumentations as A
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

# local imports
from deepchecks.vision.nir_mixed.snake_data_module import SnakeDataModule
from deepchecks.vision.utils.image_utils import AlbumentationImageFolder


class SnakeSkewedDataModule(SnakeDataModule):
    def __init__(self, skew_class: int = -1, skew_ratio: float = 1.0, *args, **kwargs):
        self._skew_class = skew_class
        self._skew_ratio = skew_ratio
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        """
        Used to skew a specific class before doing the split
        :param stage:
        :return:
        """
        if self._skew_class >= 0:
            samples_per_class = defaultdict(list)
            for sample in self.dataset.samples:
                image, label = sample
                samples_per_class[label].append(sample)
            n_class = len(samples_per_class[self._skew_class])
            n_after = int(n_class * self._skew_ratio)
            skewed_class_indices = np.random.choice(n_class, n_after, replace=False)
            skewed_class_list = [samples_per_class[self._skew_class][i] for i in skewed_class_indices]
            samples_per_class[self._skew_class] = skewed_class_list
            new_samples = [sample for k, samples in samples_per_class.items() for sample in samples]
            self.dataset.samples = new_samples
        super().setup(stage)