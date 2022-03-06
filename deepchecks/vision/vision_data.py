# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The vision/dataset module containing the vision Dataset class and its functions."""
from collections import Counter
from copy import copy
from abc import abstractmethod
from typing import Any, List, Optional, Dict, TypeVar, Union

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_functions import ImageInfo
from deepchecks.vision.utils.transformations import add_augmentation_in_start, get_transforms_handler

logger = logging.getLogger('deepchecks')
VD = TypeVar('VD', bound='VisionData')


class VisionData:
    """VisionData represent a base task in deepchecks. It wraps PyTorch DataLoader together with model related metadata.

    The VisionData class is containing additional data and general methods intended for easily accessing
    metadata relevant for validating a computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. If your data loader is using IterableDataset please see note below.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    random_seed : int, default: 0
        Random seed used to generate the sample.
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.

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
    or else the callable label_formatter should be able to transform the labels to the desired format.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 transform_field: Optional[str] = 'transforms'):

        self._data_loader = data_loader
        self._data_loader = self._get_data_loader_copy()

        self._num_classes = num_classes
        self._label_map = label_map
        self._transform_field = transform_field
        self._warned_labels = set()

        try:
            self.validate_image_data(next(iter(self._data_loader)))
            self._has_images = True
        except DeepchecksValueError:
            logger.warn('batch_to_images() was not implemented, some checks will not run')
            self._has_images = False

        self._n_of_samples_per_class = None
        self._task_type = None
        self._has_label = None
        self._current_seed = None

    @abstractmethod
    def get_classes(self, batch_labels: Union[List[torch.Tensor], torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        return NotImplementedError("get_classes() must be implemented in a subclass")

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        raise DeepchecksValueError(
            "batch_to_labels() must be implemented in a subclass"
        )

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        raise DeepchecksValueError(
            "infer_on_batch() must be implemented in a subclass"
        )

    @abstractmethod
    def validate_label(self, batch):
        raise DeepchecksValueError(
            "validate_label() must be implemented in a subclass"
        )

    @abstractmethod
    def validate_prediction(self, model, device):
        raise DeepchecksValueError(
            "validate_prediction() must be implemented in a subclass"
        )

    @abstractmethod
    def batch_to_images(self, batch) -> List[np.ndarray]:
        """Infer on batch.
        Examples
        --------
        >>> return batch[0]
        """
        raise DeepchecksValueError(
            "batch_to_images() must be implemented in a subclass"
        )

    @property
    def n_of_samples_per_class(self) -> Dict[Any, int]:
        """Return a dictionary containing the number of samples per class."""
        if self._n_of_samples_per_class is None:
            self._n_of_samples_per_class = self._get_samples_per_class()
        return copy(self._n_of_samples_per_class)

    @property
    def data_loader(self) -> DataLoader:
        """Return the data loader."""
        return self._data_loader

    @property
    def transform_field(self) -> str:
        """Return the data loader."""
        return self._transform_field

    @property
    def task_type(self) -> int:
        """Return the task type."""
        return self._task_type

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            self._num_classes = len(self.n_of_samples_per_class.keys())
        return self._num_classes

    @property
    def data_dimension(self):
        """Return how many dimensions the image data have."""
        image = self.batch_to_images(next(iter(self)))[0]  # pylint: disable=not-callable
        return ImageInfo(image).get_dimension()

    def label_id_to_name(self, class_id: int) -> str:
        """Return the name of the class with the given id."""
        # Converting the class_id to integer to make sure it is an integer
        class_id = int(class_id)

        if self._label_map is None:
            return str(class_id)
        elif class_id not in self._label_map:
            if class_id not in self._warned_labels:
                # We want to warn one time per class
                self._warned_labels.add(class_id)
                logger.warning('Class id %s is not in the label map.', class_id)
            return str(class_id)
        else:
            return self._label_map[class_id]

    def get_transform_type(self):
        """Return transforms handler created from the transform field."""
        dataset_ref = self._data_loader.dataset
        # If no field exists raise error
        if not hasattr(dataset_ref, self._transform_field):
            msg = f'Underlying Dataset instance does not contain "{self._transform_field}" attribute. If your ' \
                  f'transformations field is named otherwise, you cat set it by using "transform_field" parameter'
            raise DeepchecksValueError(msg)
        transform = dataset_ref.__getattribute__(self._transform_field)
        return get_transforms_handler(transform)

    def get_augmented_dataset(self, aug) -> VD:
        """Validate transform field in the dataset, and return a copy of the vision data object
        with the augmentation in the start of it."""
        dataset_ref = self._data_loader.dataset
        print(dataset_ref)
        # If no field exists raise error
        if not hasattr(dataset_ref, self._transform_field):
            msg = f'Underlying Dataset instance does not contain "{self._transform_field}" attribute. If your ' \
                  f'transformations field is named otherwise, you cat set it by using "transform_field" parameter'
            raise DeepchecksValueError(msg)
        transform = dataset_ref.__getattribute__(self._transform_field)
        new_transform = add_augmentation_in_start(aug, transform)
        dataset_copy = copy(dataset_ref)
        dataset_copy.__setattr__(self._transform_field, new_transform)
        new_vision_data = self.copy(dataset_copy)
        print(new_vision_data.data_loader.dataset)
        return new_vision_data

    def copy(self, new_dataset = None) -> VD:
        """Create new copy of this object, with the data-loader and dataset also copied."""
        new_data_loader = self._get_data_loader_copy(new_dataset)
        new_vision_data =  self.__class__(new_data_loader,
                                          num_classes=self.num_classes,
                                          label_map=self._label_map,
                                          transform_field=self._transform_field)
        if self._current_seed is not None:
            new_vision_data.set_seed(self._current_seed)
        return new_vision_data

    def to_batch(self, *samples):
        """Use the defined collate_fn to transform a few data items to batch format."""
        return self._data_loader.collate_fn(list(samples))

    def set_seed(self, seed: int):
        """Set seed for data loader."""
        generator: torch.Generator = self._data_loader.generator
        if generator is not None and seed is not None:
            generator.set_state(torch.Generator().manual_seed(seed).get_state())
            self._current_seed = seed

    def validate_shared_label(self, other: VD):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Parameters
        ----------
        other : VisionData
            Expected to be Dataset type. dataset to compare

        Raises
        ------
        DeepchecksValueError
            if datasets don't have the same label
        """
        if not isinstance(other, VisionData):
            raise DeepchecksValueError('Check requires dataset to be of type VisionTask. instead got: '
                                       f'{type(other).__name__}')

        if self._has_label != other._has_label:
            raise DeepchecksValueError('Datasets required to both either have or don\'t have labels')

        if self._task_type != other._task_type:
            raise DeepchecksValueError('Datasets required to have same label type')

    def validate_image_data(self, batch):
        """Validate that the data is in the required format.

        The validation is done on the first element of the batch.

        Parameters
        ----------
        batch

        Raises
        -------
        DeepchecksValueError
            If the batch data doesn't fit the format after being transformed by self().

        """
        data = self.batch_to_images(batch)
        try:
            sample: np.ndarray = data[0]
        except TypeError as err:
            raise DeepchecksValueError('The batch data must be an iterable.') from err
        if not isinstance(sample, np.ndarray):
            raise DeepchecksValueError('The data inside the iterable must be a numpy array.')
        if sample.ndim != 3:
            raise DeepchecksValueError('The data inside the iterable must be a 3D array.')
        if sample.shape[2] not in [1, 3]:
            raise DeepchecksValueError('The data inside the iterable must have 1 or 3 channels.')
        if sample.min() < 0 or sample.max() > 255:
            raise DeepchecksValueError('The data inside the iterable must be in the range [0, 255].')
        if np.all(sample <= 1):
            raise DeepchecksValueError('The data inside the iterable appear to be normalized.')

    def _get_samples_per_class(self):
        """
        Get the number of samples per class.
        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to get the samples per class from.
        Returns
        -------
        Counter
            Counter of the number of samples per class.
        """
        counter = Counter()
        for batch in self:
            labels = self.batch_to_labels(batch)
            counter.update(self.get_classes(labels))
        return counter

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data_loader)

    def __len__(self):
        """Return the number of batches in the dataset dataloader."""
        return len(self._data_loader)

    def _get_data_loader_copy(self, new_dataset = None):
        props = {
            'num_workers': self._data_loader.num_workers,
            'collate_fn': self._data_loader.collate_fn,
            'pin_memory': self._data_loader.pin_memory,
            'timeout': self._data_loader.timeout,
            'worker_init_fn': self._data_loader.worker_init_fn,
            'prefetch_factor': self._data_loader.prefetch_factor,
            'persistent_workers': self._data_loader.persistent_workers,
            'generator': torch.Generator()
        }
        # Add batch sampler if exists, else sampler
        if self._data_loader.batch_sampler is not None:
            # Can't deepcopy since generator is not pickle-able, so copying shallowly and then copies also sampler inside
            batch_sampler = copy(self._data_loader.batch_sampler)
            batch_sampler.sampler = copy(batch_sampler.sampler)
            # Replace generator instance so the copied dataset will not affect the original
            batch_sampler.sampler.generator = props['generator']
            props['batch_sampler'] = batch_sampler
        else:
            sampler = copy(self._data_loader.sampler)
            # Replace generator instance so the copied dataset will not affect the original
            sampler.generator = props['generator']
            props['sampler'] = sampler
        if new_dataset is None:
            props['dataset'] = copy(self._data_loader.dataset)
        else:
            print('added')
            props['dataset'] = copy(new_dataset)
        return self._data_loader.__class__(**props)
