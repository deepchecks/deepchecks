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
from enum import Enum
from typing import Any, Iterable, List, Optional, Dict, TypeVar, Union

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksNotImplementedError, DeepchecksValueError, ValidationError
from deepchecks.vision.utils.image_functions import ImageInfo
from deepchecks.vision.utils.transformations import add_augmentation_in_start, get_transforms_handler

logger = logging.getLogger('deepchecks')
VD = TypeVar('VD', bound='VisionData')

__all__ = ['TaskType', 'VisionData']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'


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
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 transform_field: Optional[str] = 'transforms'):

        self._data_loader = self._get_data_loader_copy(data_loader)

        self._num_classes = num_classes
        self._label_map = label_map
        self._transform_field = transform_field
        self._warned_labels = set()
        self._has_images = False

        try:
            self.validate_image_data(next(iter(self._data_loader)))
            self._has_images = True
        except DeepchecksNotImplementedError:
            logger.warning('batch_to_images() was not implemented, some checks will not run')
        except ValidationError as ex:
            logger.warning('batch_to_images() was not implemented correctly, '
                           'the validiation has failed with the error: %s', {str(ex)})

        self._n_of_samples_per_class = None
        self._task_type = None
        self._has_label = None
        self._current_seed = None

    @abstractmethod
    def get_classes(self, batch_labels: Union[List[torch.Tensor], torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        return NotImplementedError('get_classes() must be implemented in a subclass')

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        """Transform a batch of data to labels."""
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        """Infer on a batch of data."""
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    @abstractmethod
    def validate_label(self, batch):
        """Validate a batch of labels."""
        raise NotImplementedError('validate_label() must be implemented in a subclass')

    @abstractmethod
    def validate_prediction(self, batch, model, device):
        """Validate a batch of predictions."""
        raise DeepchecksValueError(
            'validate_prediction() must be implemented in a subclass'
        )

    @abstractmethod
    def batch_to_images(self, batch) -> Iterable[np.ndarray]:
        """
        Transform a batch of data to images in the accpeted format.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of data to transform to images.

        Returns
        -------
        Iterable[np.ndarray]
            List of images in the accepted format. Each image in the iterable must be a [H, W, C] 3D numpy array.
            See notes for more details.
            :func: `batch_to_images` must be implemented in a subclass.

        Examples
        --------
        >>> def batch_to_images(self, batch):
        ...     return batch[0]

        Notes
        -----
        Each image in the iterable must be a [H, W, C] 3D numpy array. The first dimension must be the image height
        (y axis), the second being the image width (x axis), and the third being the number of channels. The numbers in
        the array should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while
        grayscale images should have 1 channel.
        """
        raise DeepchecksNotImplementedError('batch_to_images() must be implemented in a subclass')

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
    def has_label(self) -> bool:
        """Return True if the data loader has labels."""
        return self._has_label

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
        """Return a copy of the vision data object with the augmentation in the start of it."""
        dataset_ref = self._data_loader.dataset
        # If no field exists raise error
        if not hasattr(dataset_ref, self._transform_field):
            msg = f'Underlying Dataset instance does not contain "{self._transform_field}" attribute. If your ' \
                  f'transformations field is named otherwise, you cat set it by using "transform_field" parameter'
            raise DeepchecksValueError(msg)
        new_vision_data = self.copy()
        new_dataset_ref = new_vision_data.data_loader.dataset
        transform = new_dataset_ref.__getattribute__(self._transform_field)
        new_transform = add_augmentation_in_start(aug, transform)
        new_dataset_ref.__setattr__(self._transform_field, new_transform)
        return new_vision_data

    def copy(self) -> VD:
        """Create new copy of this object, with the data-loader and dataset also copied."""
        new_vision_data = self.__class__(self._data_loader,
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
            raise ValidationError('Check requires dataset to be of type VisionTask. instead got: '
                                  f'{type(other).__name__}')

        if self._has_label != other.has_label:
            raise ValidationError('Datasets required to both either have or don\'t have labels')

        if self._task_type != other.task_type:
            raise ValidationError('Datasets required to have same label type')

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
            raise ValidationError('The batch data must be an iterable.') from err
        if not isinstance(sample, np.ndarray):
            raise ValidationError('The data inside the iterable must be a numpy array.')
        if sample.ndim != 3:
            raise ValidationError('The data inside the iterable must be a 3D array.')
        if sample.shape[2] not in [1, 3]:
            raise ValidationError('The data inside the iterable must have 1 or 3 channels.')
        if sample.min() < 0 or sample.max() > 255:
            raise ValidationError('The data inside the iterable must be in the range [0, 255].')
        if np.all(sample <= 1):
            raise ValidationError('The data inside the iterable appear to be normalized.')

    def _get_samples_per_class(self):
        """
        Get the number of samples per class.

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

    @staticmethod
    def _get_data_loader_copy(data_loader: DataLoader):
        props = {
            'num_workers': data_loader.num_workers,
            'collate_fn': data_loader.collate_fn,
            'pin_memory': data_loader.pin_memory,
            'timeout': data_loader.timeout,
            'worker_init_fn': data_loader.worker_init_fn,
            'prefetch_factor': data_loader.prefetch_factor,
            'persistent_workers': data_loader.persistent_workers,
            'generator': torch.Generator()
        }
        # Add batch sampler if exists, else sampler
        if data_loader.batch_sampler is not None:
            # Can't deepcopy since generator is not pickle-able,
            # so copying shallowly and then copies also sampler inside
            batch_sampler = copy(data_loader.batch_sampler)
            batch_sampler.sampler = copy(batch_sampler.sampler)
            # Replace generator instance so the copied dataset will not affect the original
            batch_sampler.sampler.generator = props['generator']
            props['batch_sampler'] = batch_sampler
        else:
            sampler = copy(data_loader.sampler)
            # Replace generator instance so the copied dataset will not affect the original
            sampler.generator = props['generator']
            props['sampler'] = sampler
        props['dataset'] = copy(data_loader.dataset)
        return data_loader.__class__(**props)
