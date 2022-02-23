import os
import shutil
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
    """
    This is a LightningDataModule subclass.
    This wraps a Dataset object and takes care of splitting, if requires.
    setup() is ran when training begins and does splits.

    Additionally, this can be either given a "dataset" - and then split happens.
    It can alternatively receive a train and val directories, in which case the datasets are used as is.
    """
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
        self.train, self.val, remainder = random_split(self.dataset, [n_train, n_val, len(self.dataset) - n])
        self.train.dataset.transforms = self._train_transforms
        self.val.dataset.transforms = self._val_transforms
        self.save_data_partitions()

    def save_data_partitions(self, output_dir: Optional[str] = None,
                             copy: Optional[bool] = True):
        """
        This method is called externally to save partitions - in case they where created by the
        Module (or simply handed them).
        This allows for experiment reusability.
        If provided 'copy' is True, the data is actually copied.
        Otherwise, only train/val txt lists are created.
        :param output_dir:
        :param copy:
        :return:
        """
        if not self.train:
            print("Partitions not created yet!", file=sys.stderr)
            return
        output_dir = output_dir or os.path.join(self.data_dir, "splits")
        self.save_dataset(self.dataset, self.train, dataset_name="train",
                          output_dir=output_dir, ref_dir=self.data_dir,
                          copy_files=copy)
        self.save_dataset(self.dataset, self.val, dataset_name="val",
                          output_dir=output_dir, ref_dir=self.data_dir,
                          copy_files=copy)

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
        return self.predict_dataloader()

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

    @staticmethod
    def save_dataset(full_dataset, subset, output_dir, dataset_name, copy_files=True, ref_dir=""):
        """Aux method for saving the splits."""
        dataset_file = f"{dataset_name}.txt"
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        data_list = []
        for subset_idx in subset.indices:
            sample = full_dataset.samples[subset_idx]
            img_path, label = sample
            if copy_files:
                target_dir = os.path.join(dataset_dir, f"class-{str(label)}")
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(img_path, target_dir)

            img_path = os.path.relpath(img_path, ref_dir)
            row_str = f"{img_path}\t{str(label)}\n"
            data_list.append(row_str)

        with open(os.path.join(output_dir, dataset_file), 'w') as fid:
            [fid.write(row_str) for row_str in data_list]