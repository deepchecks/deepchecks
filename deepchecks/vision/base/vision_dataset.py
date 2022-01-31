from copy import copy
from enum import Enum
from collections import Counter
from typing import Optional

import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch import cat
import logging

from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.typing import Hashable
from deepchecks.vision.utils.image_transforms import UnNormalize, ReverseToTensorV2

logger = logging.getLogger('deepchecks')

__all__ = ['TaskType', 'VisionDataset']


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


class VisionDataset:

    _data: DataLoader = None

    def __init__(self, data_loader: Optional[DataLoader],
                 num_classes: Optional[int] = None,
                 label_type: Optional[str] = None,
                 transform_field: Optional[str] = "transform"):
        self._transform_field = transform_field
        self._data = data_loader
        if label_type is not None:
            self.label_type = label_type
        else:
            self.label_type = self.infer_label_type()

        self._num_classes = num_classes  # if not initialized, then initialized later in get_num_classes()
        self._samples_per_class = None
        self._inverse_transform = None

    @property
    def num_classes(self):
        return self.get_num_classes()

    @property
    def samples_per_class(self):
        return self.get_samples_per_class()

    @property
    def label_shape(self):
        return self.get_label_shape()

    @property
    def data_loader(self):
        return self.get_data_loader()

    def get_num_classes(self):
        if self._num_classes is None:
            samples_per_class = self.get_samples_per_class()
            num_classes = len(samples_per_class.keys())
            self._num_classes = num_classes
        return self._num_classes

    def get_samples_per_class(self):
        if self._samples_per_class is None:
            if self.label_type == TaskType.CLASSIFICATION.value:
                counter = Counter()
                for i in range(len(self._data)):
                    counter.update(next(iter(self._data))[1].tolist())
                self._samples_per_class = counter
            elif self.label_type == TaskType.OBJECT_DETECTION.value:
                # Assume next(iter(self._data))[1] is a list (per sample) of numpy arrays (rows are bboxes) with the
                # first column in the array representing class
                counter = Counter()
                for i in range(len(self._data)):
                    list_of_arrays = next(iter(self._data))[1]
                    class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
                    counter.update(class_list)
                self._samples_per_class = counter
            else:
                raise NotImplementedError(
                    'Not implemented yet for tasks other than classification and object detection'
                )
        return copy(self._samples_per_class)

    def extract_label(self):
        y = []
        for i in range(len(self._data)):
            y.append(next(iter(self._data))[1])
        return cat(y, 0)

    def infer_label_type(self):
        label_shape = self.get_label_shape()

        # Means the tensor is an array of scalars
        if len(label_shape) == 0:
            return TaskType.CLASSIFICATION.value
        else:
            return TaskType.OBJECT_DETECTION.value

    def validate_label(self):
        # Getting first sample of data
        sample = self._data.dataset[0]
        if len(sample) != 2:
            raise DeepchecksValueError('Check requires dataset to have a label')

    def get_label_shape(self):
        self.validate_label()

        # Assuming the dataset contains a tuple of (features, label)
        return next(iter(self._data))[1][0].shape  # first argument is batch_size

    def __iter__(self):
        return iter(self._data)

    def get_data_loader(self):
        return self._data

    def validate_shared_label(self, other):
        """Verify presence of shared labels.

        Validates whether the 2 datasets share the same label shape

        Args:
            other (Dataset): Expected to be Dataset type. dataset to compare

        Returns:
            Hashable: name of the label column

        Raises:
            DeepchecksValueError if datasets don't have the same label
        """
        VisionDataset.validate_dataset(other)

        label_shape = self.get_label_shape()
        other_label_shape = other.get_label_shape()

        if other_label_shape != label_shape:
            raise DeepchecksValueError('Check requires datasets to share the same label shape')

    @classmethod
    def validate_dataset(cls, obj) -> 'VisionDataset':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Args:
            obj: object to validate as dataset

        Returns:
            (Dataset): object that is deepchecks dataset
        """
        if not isinstance(obj, VisionDataset):
            raise DeepchecksValueError('Check requires dataset to be of type VisionDataset. instead got: '
                                       f'{type(obj).__name__}')
        if len(obj._data.dataset) == 0:
            raise DeepchecksValueError('Check requires a non-empty dataset')

        return obj

    def validate_transforms(self):
        """I
        This checks that a field of name "transform" exists as it should
        Definitely needs expanding for more dataset support
        :return:
        """
        import albumentations as A
        dataset_ref = self.get_data_loader().dataset
        try:
            transform_field = dataset_ref.__getattribute__(self._transform_field)
            # If a list, create A.Compose out of it; shouldn't really happen though
            if isinstance(transform_field, list):
                dataset_ref.__setattr__(self._transform_field,
                                        A.Compose(transform_field))
            # Otherwise if it's not an albumentations object, throw exception
            elif not isinstance(transform_field, A.Compose):
                raise DeepchecksValueError("Dataset.transform field must be of instance type albumentations.Compose")
        # If no field exists this is another issue
        except AttributeError as e:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self._transform_field} attribute")

    # TODO move two methods inside VisionDataset
    def add_dataset_transforms(self, op: A.BasicTransform = A.NoOp):
        try:
            dataset_ref = self.get_data_loader().dataset
            transform_object = dataset_ref.__getattribute__(self._transform_field)
            dataset_ref.__setattr__(self._transform_field, A.Compose([op()] + transform_object.transforms.transforms))
        except AttributeError as e:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self._transform_field} attribute")

    def edit_dataset_transforms(self, op: A.BasicTransform = A.NoOp, idx: int = 0):
        try:
            dataset_ref = self.get_data_loader().dataset
            transform_object = dataset_ref.__getattribute__(self._transform_field)
            dataset_ref.__setattr__(self._transform_field, A.Compose([op] + transform_object.transforms.transforms[1:]))
        except AttributeError as e:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self._transform_field} attribute")

    def inverse_transform(self, sample: torch.Tensor):
        """
        # TODO perhaps use like a property?
        :param sample:
        :return:
        """
        if self._inverse_transform is None:
            self._inverse_transform = self._create_inverse_transform()
        return self._inverse_transform.apply(image=sample)["image"]

    def _create_inverse_transform(self):
        import albumentations as A
        aug_types = [type(a) for a in self.get_transforms()]
        # Find Normalize
        final_augmentations = []
        try:
            norm_idx = aug_types.index(A.Normalize)
            final_augmentations.append(UnNormalize(self.get_transforms()[norm_idx]))
        except ValueError:
            pass
        try:
            totensor_idx = aug_types.index(A.pytorch.ToTensorV2)
            final_augmentations.append(ReverseToTensorV2(self.get_transforms()[totensor_idx]))
        except ValueError:
            pass
        return A.Compose(list(reversed(final_augmentations)))


    def get_transforms(self):
        try:
            dataset_ref = self.get_data_loader().dataset
            transform_object = dataset_ref.__getattribute__(self._transform_field)
            return transform_object
        except AttributeError as e:
            raise DeepchecksValueError(f"Underlying Dataset instance must have a {self._transform_field} attribute")

