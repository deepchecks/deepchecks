import os.path as osp
import sys
from typing import Optional
import albumentations as A
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

# local imports
from deepchecks.vision.utils.image_utils import AlbumentationImageFolder


class SnakeDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: Optional[str] = None,
                 train_data_dir: Optional[str] = None,
                 val_data_dir: Optional[str] = None,
                 train_transforms: Optional[A.Compose] = A.Compose([A.NoOp]),
                 val_transforms: Optional[A.Compose] = A.Compose([A.NoOp]),
                 batch_size: Optional[int] = 128,
                 num_workers: Optional[int] = 8):
        super().__init__()
        self.data_dir = data_dir
        no_train_val_split = train_data_dir is None and val_data_dir is None
        assert (no_train_val_split and data_dir is not None) or \
               not no_train_val_split and data_dir is None
        if data_dir:
            self.dataset = AlbumentationImageFolder(root=data_dir)
            self.train = None
            self.val = None
        else:
            self.train = AlbumentationImageFolder(root=train_data_dir)
            self.dataset = None
            self.val = AlbumentationImageFolder(root=val_data_dir)
        self._batch_size = batch_size
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        no_train_val_split = self.train is None and self.val is None
        if no_train_val_split:
            self._train_val_split()
        else:
            self.train.transforms = self._train_transforms
            self.val.transforms = self._val_transforms

    def _train_val_split(self):
        """
        Splits train and validation datasets for training, if only a single dataset was given to init
        If both were given this only assigns the transforms
        :return:
        """
        n = len(self.dataset)
        n_train = int(np.ceil(0.8 * n))
        n_val = int(np.floor(0.2 * n))
        self.train, self.val = random_split(self.dataset, [n_train, n_val])
        self.train.dataset.transforms = self._train_transforms
        self.val.dataset.transforms = self._val_transforms

    @property
    def num_classes(self):
        if self.dataset is None:
            return len(self.val.classes)
        return len(self.dataset.classes)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self._batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        """
        For testing we make reuse of the validation set
        :return:
        """
        return self.val_dataloader()

    def predict_dataloader(self):
        """
        Note that for the simplicity and the reusability of this Lightning Data Module,
         we assume the entire dataset is the one used for prediction
        :return:
        """
        if self.dataset is None:
            raise RuntimeError("Module isn't initialized for prediction but for train/val.")
        self.dataset.transforms = self._val_transforms
        return DataLoader(self.dataset, batch_size=self._batch_size, num_workers=self._num_workers)