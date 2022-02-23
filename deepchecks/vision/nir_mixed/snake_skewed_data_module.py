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
    def __init__(self, skew_class: int = -1, skew_ratio: float = 1.0,
                 subset_size: Optional[int] = 0,
                 *args, **kwargs):
        self._skew_class = skew_class
        self._skew_ratio = skew_ratio
        self._subset_size = subset_size
        super().__init__(*args, **kwargs)

    def _train_val_split(self):
        """
        Splits train and validation datasets for training, if only a single dataset was given to init
        If both were given this only assigns the transforms
        :return:
        """
        if self._subset_size:
            n = self._subset_size
        else:
            n = len(self.dataset)
        n_train = int(np.ceil(0.8 * n))
        n_val = int(np.floor(0.2 * n))
        self.train, self.val, remainder = random_split(self.dataset, [n_train, n_val, len(self.dataset) - n])
        self.train.dataset.transforms = self._train_transforms
        self.val.dataset.transforms = self._val_transforms

        def skew_subset(subset):
            samples_per_class = defaultdict(list)
            for subset_idx in subset.indices:
                sample = self.dataset.samples[subset_idx]
                image, label = sample
                samples_per_class[label].append(subset_idx)
            n_class = len(samples_per_class[self._skew_class])
            n_after = int(n_class * self._skew_ratio)
            skewed_class_indices = np.random.choice(samples_per_class[self._skew_class], n_after, replace=False)
            samples_per_class[self._skew_class] = skewed_class_indices
            new_indices = [index for k, indices in samples_per_class.items() for index in indices]
            subset.indices = new_indices

        if self._skew_class >= 0:
            skew_subset(self.train)

        self.save_data_partitions()