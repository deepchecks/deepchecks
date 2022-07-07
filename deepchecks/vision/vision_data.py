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
# pylint: disable=protected-access
import random
from abc import abstractmethod
from collections import defaultdict
from copy import copy
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from deepchecks.core.errors import (DeepchecksBaseError, DeepchecksNotImplementedError, DeepchecksValueError,
                                    ValidationError)
from deepchecks.utils.logger import get_logger
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.image_functions import ImageInfo
from deepchecks.vision.utils.transformations import get_transforms_handler

__all__ = ['VisionData']


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
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        num_classes: Optional[int] = None,
        label_map: Optional[Dict[int, str]] = None,
        transform_field: Optional[str] = 'transforms'
    ):
        # Create data loader that uses IndicesSequentialSampler, which always return batches in the same order
        self._data_loader, self._sampler = self._get_data_loader_sequential(data_loader)

        self._num_classes = num_classes
        self._label_map = label_map
        self._transform_field = transform_field
        self._image_formatter_error = None
        self._label_formatter_error = None
        self._get_classes_error = None

        batch = next(iter(self._data_loader))
        try:
            self.validate_image_data(batch)
        except DeepchecksNotImplementedError:
            self._image_formatter_error = 'batch_to_images() was not implemented, some checks will not run'
            get_logger().warning(self._image_formatter_error)
        except ValidationError as ex:
            self._image_formatter_error = f'batch_to_images() was not implemented correctly, the validation has ' \
                                          f'failed with the error: "{ex}". To test your image formatting use the ' \
                                          f'function `validate_image_data(batch)`'
            get_logger().warning(self._image_formatter_error)

        try:
            self.validate_label(batch)
        except DeepchecksNotImplementedError:
            self._label_formatter_error = 'batch_to_labels() was not implemented, some checks will not run'
            get_logger().warning(self._label_formatter_error)
        except ValidationError as ex:
            self._label_formatter_error = f'batch_to_labels() was not implemented correctly, the validation has ' \
                                          f'failed with the error: "{ex}". To test your label formatting use the ' \
                                          f'function `validate_label(batch)`'
            get_logger().warning(self._label_formatter_error)

        try:
            if self._label_formatter_error is None:
                self.validate_get_classes(batch)
            else:
                self._get_classes_error = 'Must have valid labels formatter to use `get_classes`'
        except DeepchecksNotImplementedError:
            self._get_classes_error = 'get_classes() was not implemented, some checks will not run'
            get_logger().warning(self._get_classes_error)
        except ValidationError as ex:
            self._get_classes_error = f'get_classes() was not implemented correctly, the validation has ' \
                                      f'failed with the error: "{ex}". To test your formatting use the ' \
                                      f'function `validate_get_classes(batch)`'
            get_logger().warning(self._get_classes_error)

        self._classes_indices = None
        self._current_index = None

    @classmethod
    def from_dataset(
        cls: Type[VD],
        data: Dataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        num_classes: Optional[int] = None,
        label_map: Optional[Dict[int, str]] = None,
        transform_field: Optional[str] = 'transforms'
    ) -> VD:
        """Create VisionData instance from a Dataset instance.

        Parameters
        ----------
        data : Dataset
            instance of a Dataset.
        batch_size: int, default 64
            how many samples per batch to load.
        shuffle : bool, default True:
            set to ``True`` to have the data reshuffled at every epoch.
        num_workers  int, default 0:
            how many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process.
        pin_memory bool, default True
            If ``True``, the data loader will copy Tensors into CUDA pinned memory
            before returning them.
        collate_fn : Optional[Callable]
            merges a list of samples to form a mini-batch of Tensor(s).
        num_classes : Optional[int], default None
            Number of classes in the dataset. If not provided, will be inferred from the dataset.
        label_map : Optional[Dict[int, str]], default None
            A dictionary mapping class ids to their names.
        transform_field : Optional[str], default: 'transforms'
            Name of transforms field in the dataset which holds transformations of both data and label.

        Returns
        -------
        VisionData
        """
        def batch_collate(batch):
            imgs, labels = zip(*batch)
            return list(imgs), list(labels)

        return cls(
            data_loader=DataLoader(
                dataset=data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn or batch_collate
            ),
            num_classes=num_classes,
            label_map=label_map,
            transform_field=transform_field
        )

    @abstractmethod
    def get_classes(self, batch_labels: Union[List[torch.Tensor], torch.Tensor]) -> List[List[int]]:
        """Get a labels batch and return classes inside it."""
        raise DeepchecksNotImplementedError('get_classes() must be implemented in a subclass')

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        """Transform a batch of data to labels."""
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        """Infer on a batch of data."""
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    @abstractmethod
    def batch_to_images(self, batch) -> Sequence[np.ndarray]:
        """
        Transform a batch of data to images in the accpeted format.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of data to transform to images.

        Returns
        -------
        Sequence[np.ndarray]
            List of images in the accepted format. Each image in the iterable must be a [H, W, C] 3D numpy array.
            See notes for more details.
            :func: `batch_to_images` must be implemented in a subclass.

        Examples
        --------
        >>> import numpy as np
        ...
        ...
        ... def batch_to_images(self, batch):
        ...     # Converts a batch of normalized images to rgb images with range [0, 255]
        ...     inp = batch[0].detach().numpy().transpose((0, 2, 3, 1))
        ...     mean = [0.485, 0.456, 0.406]
        ...     std = [0.229, 0.224, 0.225]
        ...     inp = std * inp + mean
        ...     inp = np.clip(inp, 0, 1)
        ...     return inp * 255

        Notes
        -----
        Each image in the iterable must be a [H, W, C] 3D numpy array. The first dimension must be the image height
        (y axis), the second being the image width (x axis), and the third being the number of channels. The numbers in
        the array should be in the range [0, 255]. Color images should be in RGB format and have 3 channels, while
        grayscale images should have 1 channel.
        """
        raise DeepchecksNotImplementedError('batch_to_images() must be implemented in a subclass')

    def validate_label(self, batch):
        """Validate a batch of labels."""
        # default implementation just calling the function to see if it runs
        self.batch_to_labels(batch)

    def validate_prediction(self, batch, model, device):
        """
        Validate the prediction.

        Parameters
        ----------
        batch : t.Any
            Batch from DataLoader
        model : t.Any
        device : torch.Device

        Raises
        ------
        ValidationError
            If predictions format is invalid (depends on validate_infered_batch_predictions implementations)
        DeepchecksNotImplementedError
            If infer_on_batch not implemented
        """
        self.validate_infered_batch_predictions(self.infer_on_batch(batch, model, device))

    @staticmethod
    def validate_infered_batch_predictions(batch_predictions):
        """Validate the infered predictions from the batch."""
        # isn't relevant for this class but is still a function we want to inherit

    def update_cache(self, batch: Batch):
        """Get labels and update the classes' metadata info."""
        try:
            # In case there are no labels or there is an invalid formatter function, this call will raise exception
            classes_per_label = self.get_classes(batch.labels)
        except DeepchecksBaseError:
            self._classes_indices = None
            return

        for batch_index, classes in enumerate(classes_per_label):
            for single_class in classes:
                dataset_index = self.to_dataset_index(self._current_index + batch_index)[0]
                self._classes_indices[single_class].append(dataset_index)
        self._current_index += len(classes_per_label)

    def init_cache(self):
        """Initialize the cache of the classes' metadata info."""
        self._classes_indices = defaultdict(list)
        self._current_index = 0

    @property
    def classes_indices(self) -> Dict[int, List[int]]:
        """Return dict of classes as keys, and list of corresponding indices (in Dataset) of samples that include this\
        class (in the label)."""
        if self._classes_indices is None:
            raise DeepchecksValueError('Could not process labels.')
        if self._current_index < len(self._sampler):
            raise DeepchecksValueError('Cached data is not computed on all the data yet.')
        return self._classes_indices

    @property
    def n_of_samples_per_class(self) -> Dict[Any, int]:
        """Return a dictionary containing the number of samples per class."""
        return {k: len(v) for k, v in self.classes_indices.items()}

    @property
    def data_loader(self) -> DataLoader:
        """Return the data loader."""
        return self._data_loader

    @property
    def transform_field(self) -> str:
        """Return the data loader."""
        return self._transform_field

    @property
    def has_labels(self) -> bool:
        """Return True if the data loader has labels."""
        return self._label_formatter_error is None

    @property
    def has_images(self) -> bool:
        """Return True if the data loader has images."""
        return self._image_formatter_error is None

    @property
    def task_type(self) -> TaskType:
        """Return the task type: classification, object_detection or other."""
        return TaskType.OTHER

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            self._num_classes = len(self.classes_indices.keys())
        return self._num_classes

    @property
    def num_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._sampler)

    @property
    def original_num_samples(self) -> int:
        """Return the number of samples in the original dataset."""
        return len(self._data_loader.dataset)

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
            get_logger().warning('Class id %s is not in the label map. Add it to map '
                                 'in order to show the class name instead of id', class_id)
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
        return get_transforms_handler(transform, self.task_type)

    def get_augmented_dataset(self, aug) -> VD:
        """Return a copy of the vision data object with the augmentation in the start of it."""
        transform_handler = self.get_transform_type()
        new_vision_data = self.copy()
        new_dataset_ref = new_vision_data.data_loader.dataset
        transform = new_dataset_ref.__getattribute__(self._transform_field)
        new_transform = transform_handler.add_augmentation_in_start(aug, transform)
        new_dataset_ref.__setattr__(self._transform_field, new_transform)
        return new_vision_data

    def copy(self, n_samples: int = None, shuffle: bool = False, random_state: int = None) -> VD:
        """Create new copy of this object, with the data-loader and dataset also copied, and altered by the given \
        parameters.

        Parameters
        ----------
        n_samples : int , default: None
            take only this number of samples to the copied DataLoader. The samples which will be chosen are affected
            by random_state (fixed random state will return consistent samples).
        shuffle : bool, default: False
            Whether to shuffle the samples order. The shuffle is affected random_state (fixed random state will return
            consistent order)
        random_state : int , default: None
            random_state used for the psuedo-random actions (sampling and shuffling)
        """
        new_vision_data = copy(self)
        copied_data_loader, copied_sampler = self._get_data_loader_copy(
            self.data_loader, shuffle=shuffle, random_state=random_state, n_samples=n_samples
        )
        new_vision_data._data_loader = copied_data_loader
        new_vision_data._sampler = copied_sampler
        # If new data is sampled, then needs to re-calculate cache
        if n_samples and self._classes_indices is not None:
            new_vision_data.init_cache()
            for batch in new_vision_data:
                new_vision_data.update_cache(self.batch_to_labels(batch))
        return new_vision_data

    def to_batch(self, *samples):
        """Use the defined collate_fn to transform a few data items to batch format."""
        return self._data_loader.collate_fn(list(samples))

    def to_dataset_index(self, *batch_indices):
        """Return for the given batch_index the sample index in the dataset object."""
        return [self._sampler.index_at(i) for i in batch_indices]

    def batch_of_index(self, *indices):
        """Return batch samples of the given batch indices."""
        dataset_indices = self.to_dataset_index(*indices)
        samples = [self._data_loader.dataset[i] for i in dataset_indices]
        return self.to_batch(*samples)

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

        if self.has_labels != other.has_labels:
            raise ValidationError('Datasets required to both either have or don\'t have labels')

        if self.task_type != other.task_type:
            raise ValidationError('Datasets required to have same label type')

    def validate_image_data(self, batch):
        """Validate that the data is in the required format.

        The validation is done on the first element of the batch.

        Parameters
        ----------
        batch

        Raises
        ------
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
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        if sample_min < 0 or sample_max > 255 or sample_max <= 1:
            raise ValidationError(f'Image data should be in uint8 format(integers between 0 and 255). '
                                  f'Found values in range [{sample_min}, {sample_max}].')

    def validate_get_classes(self, batch):
        """Validate that the get_classes function returns data in the correct format.

        Parameters
        ----------
        batch

        Raises
        ------
        ValidationError
            If the classes data doesn't fit the format after being transformed.
        """
        class_ids = self.get_classes(self.batch_to_labels(batch))
        if not isinstance(class_ids, Sequence):
            raise ValidationError('The classes must be a sequence.')
        if not all((isinstance(x, Sequence) for x in class_ids)):
            raise ValidationError('The classes sequence must contain as values sequences of ints '
                                  '(sequence per sample).')
        if not all((all((isinstance(x, int) for x in inner_ids)) for inner_ids in class_ids)):
            raise ValidationError('The samples sequence must contain only int values.')

    def validate_format(self, model, device=None):
        """Validate the correctness of the data class implementation according to the expected format.

        Parameters
        ----------
        model : Model
            Model to validate the data class implementation against.
        device
            Device to run the model on.
        """
        from deepchecks.vision.utils.validation import validate_extractors  # pylint: disable=import-outside-toplevel
        validate_extractors(self, model, device=device)

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data_loader)

    def __len__(self):
        """Return the number of batches in the dataset dataloader."""
        return len(self._data_loader)

    def is_sampled(self):
        """Return whether the vision data is running on sample of the data."""
        return self.num_samples < self.original_num_samples

    def assert_images_valid(self):
        """Assert the image formatter defined is valid. Else raise exception."""
        if self._image_formatter_error is not None:
            raise DeepchecksValueError(self._image_formatter_error)

    def assert_labels_valid(self):
        """Assert the label formatter defined is valid. Else raise exception."""
        if self._label_formatter_error is not None:
            raise DeepchecksValueError(self._label_formatter_error)
        if self._get_classes_error is not None:
            raise DeepchecksValueError(self._get_classes_error)

    @staticmethod
    def _get_data_loader_copy(data_loader: DataLoader, n_samples: int = None, shuffle: bool = False,
                              random_state: int = None):
        """Get a copy of DataLoader which is already using IndicesSequentialSampler, altered by the given parameters.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to copy
        n_samples : int , default: None
            take only this number of samples to the copied DataLoader. The samples which will be chosen are affected
            by random_state (fixed random state will return consistent sampels).
        shuffle : bool, default: False
            Whether to shuffle the samples order. The shuffle is affected random_state (fixed random state will return
            consistent order)
        random_state : int , default: None
            random_state used for the psuedo-random actions (sampling and shuffling)
        """
        # Get sampler and copy it indices if it's already IndicesSequentialSampler
        batch_sampler = data_loader.batch_sampler
        if isinstance(batch_sampler.sampler, IndicesSequentialSampler):
            indices = batch_sampler.sampler.indices
        else:
            raise DeepchecksValueError('Expected data loader with sampler of type IndicesSequentialSampler')
        # If got number of samples which is smaller than the number of samples we currently have,
        # then take random sample
        if n_samples and n_samples < len(batch_sampler.sampler):
            size = min(n_samples, len(indices))
            if random_state is not None:
                random.seed(random_state)
            indices = random.sample(indices, size)
        # Shuffle indices if need
        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            indices = random.sample(indices, len(indices))
        # Create new sampler and batch sampler
        sampler = IndicesSequentialSampler(indices)
        new_batch_sampler = BatchSampler(sampler, batch_sampler.batch_size, batch_sampler.drop_last)

        props = VisionData._get_data_loader_props(data_loader)
        props['batch_sampler'] = new_batch_sampler
        return data_loader.__class__(**props), sampler

    @staticmethod
    def _get_data_loader_props(data_loader: DataLoader):
        """Get properties relevant for the copy of a DataLoader."""
        attr_list = ['num_workers',
                     'collate_fn',
                     'pin_memory',
                     'timeout',
                     'worker_init_fn',
                     'prefetch_factor',
                     'persistent_workers']
        aval_attr = {}
        for attr in attr_list:
            if hasattr(data_loader, attr):
                aval_attr[attr] = getattr(data_loader, attr)
        aval_attr['dataset'] = copy(data_loader.dataset)
        return aval_attr

    @staticmethod
    def _get_data_loader_sequential(data_loader: DataLoader):
        """Create new DataLoader with sampler of type IndicesSequentialSampler. This makes the data loader have \
        consistent batches order."""
        # First set generator seed to make it reproducible
        if data_loader.generator:
            data_loader.generator.set_state(torch.Generator().manual_seed(42).get_state())
        indices = []
        batch_sampler = data_loader.batch_sampler
        # Using the batch sampler to get all indices
        for batch in batch_sampler:
            indices += batch

        # Create new sampler and batch sampler
        sampler = IndicesSequentialSampler(indices)
        new_batch_sampler = BatchSampler(sampler, batch_sampler.batch_size, batch_sampler.drop_last)

        props = VisionData._get_data_loader_props(data_loader)
        props['batch_sampler'] = new_batch_sampler
        return data_loader.__class__(**props), sampler


class IndicesSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """

    indices: List[int]

    def __init__(self, indices: List[int]) -> None:
        super().__init__(None)
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def index_at(self, location):
        """Return for a given location, the real index value."""
        return self.indices[location]
