# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The vision/dataset module containing the vision Dataset class and its functions."""

from copy import copy
from enum import Enum
from collections import Counter
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from deepchecks.core.errors import DeepchecksValueError

logger = logging.getLogger('deepchecks')

__all__ = ['TaskType', 'VisionDataset']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


class VisionDataset:
    """VisionDataset wraps a PyTorch DataLoader together with model related metadata.

    The VisionDataset class is containing additional data and methods intended for easily accessing
    metadata relevant for the training or validating of a computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object
    label_type : str
        Type of label. Must be one of the following: 'classification', 'object_detection'.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_transformer : Callable, optional
        A callable, transforming a batch of labels returned by the dataloader to a batch of labels in the desired
        format.

    Notes
    -----
    Accepted label formats are:
        * Classification: tensor of shape (N,), When N is the number of samples. Each element is an integer
          representing the class index.
        * Object Detection: List of length N containing tensors of shape (B, 5), where N is the number of samples,
          B is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
          x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
          the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.

    The labels returned by the data loader (e.g. by using next(iter(data_loader))[1]) should be in the specified format,
    or else the callable label_transformer should be able to transform the labels to the desired format.
    """

    _data: DataLoader = None

    def __init__(self, data_loader: DataLoader, label_type: str, num_classes: Optional[int] = None,
                 label_transformer: Optional[Callable] = None):
        self._data = data_loader

        if label_transformer is None:
            self.label_transformer = lambda x: x
        else:
            self.label_transformer = label_transformer

        valid_label_types = [tt.value for tt in TaskType]
        if label_type in valid_label_types:
            self.label_type = label_type
        else:
            raise DeepchecksValueError(f'Invalid label type: {label_type}, must be one of {valid_label_types}.')

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self._samples_per_class = None

    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            samples_per_class = self.get_samples_per_class()
            num_classes = len(samples_per_class.keys())
            self._num_classes = num_classes
        return self._num_classes

    def get_samples_per_class(self):
        """Return a dictionary containing the number of samples per class."""
        if self._samples_per_class is None:
            if self.label_type == TaskType.CLASSIFICATION.value:
                counter = Counter()
                for _ in range(len(self._data)):
                    counter.update(self.label_transformer(next(iter(self._data))[1].tolist()))
                self._samples_per_class = counter
            elif self.label_type == TaskType.OBJECT_DETECTION.value:
                # Assume next(iter(self._data))[1] is a list (per sample) of numpy arrays (rows are bboxes) with the
                # first column in the array representing class
                counter = Counter()
                for _ in range(len(self._data)):
                    list_of_arrays = self.label_transformer(next(iter(self._data))[1])
                    class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
                    counter.update(class_list)
                self._samples_per_class = counter
            else:
                raise NotImplementedError(
                    'Not implemented yet for tasks other than classification and object detection'
                )
        return copy(self._samples_per_class)

    def validate_label(self):
        """Validate the label type of the dataset."""
        # Getting first sample of data
        batch = next(iter(self.get_data_loader()))
        if len(batch) != 2:
            raise DeepchecksValueError('Check requires dataset to have a label')

        label_batch = self.label_transformer(batch[1])
        if self.label_type == TaskType.CLASSIFICATION.value:
            if not isinstance(label_batch, (torch.Tensor, np.ndarray)):
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a torch.Tensor or numpy '
                                           f'array')
            label_shape = label_batch.shape
            if len(label_shape) != 1:
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a 1D tensor')
        elif self.label_type == TaskType.OBJECT_DETECTION.value:
            if not isinstance(label_batch, list):
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a list with an entry for each'
                                           f' sample')
            if len(label_batch) == 0:
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a non-empty list')
            if not isinstance(label_batch[0], (torch.Tensor, np.ndarray)):
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a list of torch.Tensor or'
                                           f' numpy array')
            if len(label_batch[0].shape) != 2:
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a list of 2D tensors')
            if label_batch[0].shape[1] != 5:
                raise DeepchecksValueError(f'Check requires {self.label_type} label to be a list of 2D tensors, when '
                                           f'each row has 5 columns: [class_id, x, y, width, height]')
        else:
            raise NotImplementedError(
                'Not implemented yet for tasks other than classification and object detection'
            )

    def get_label_shape(self):
        """Return the shape of the label."""
        self.validate_label()

        # Assuming the dataset contains a tuple of (features, label)
        return self.label_transformer(next(iter(self._data))[1])[0].shape  # first argument is batch_size

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data)

    def get_data_loader(self):
        """Return the data loader."""
        return self._data

    def validate_shared_label(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Parameters
        ----------
        other : Dataset
            Expected to be Dataset type. dataset to compare
        Returns
        -------
        Hashable
            name of the label column
        Raises
        ------
        DeepchecksValueError
            if datasets don't have the same label
        """
        VisionDataset.validate_dataset(other)

        label_shape = self.get_label_shape()
        other_label_shape = other.get_label_shape()

        if other_label_shape != label_shape:
            raise DeepchecksValueError('Check requires datasets to share the same label shape')

    @classmethod
    def validate_dataset(cls, obj) -> 'VisionDataset':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Parameters
        ----------
        obj : any
            object to validate as dataset
        Returns
        -------
        Dataset
            object that is deepchecks dataset
        """
        if not isinstance(obj, VisionDataset):
            raise DeepchecksValueError('Check requires dataset to be of type VisionDataset. instead got: '
                                       f'{type(obj).__name__}')

        return obj
