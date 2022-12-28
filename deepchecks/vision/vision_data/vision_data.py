# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the VisionData class and its functions."""
import typing as t
from collections import defaultdict

from typing_extensions import Literal

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.format_validators import (validate_additional_data_format,
                                                             validate_embeddings_format,
                                                             validate_image_identifiers_format, validate_images_format,
                                                             validate_labels_format, validate_predictions_format)
from deepchecks.vision.vision_data.utils import (BatchOutputFormat, get_class_ids_from_numpy_labels,
                                                 get_class_ids_from_numpy_preds, shuffle_loader)

VD = t.TypeVar('VD', bound='VisionData')


class VisionData:
    """VisionData is the base data object of deepcheck vision used as input to checks and suites.

    Parameters
    ----------
    batch_loader :
        A batch_loader which load a batch of data in an iterative manner. Batch loader batch output must be a
        dictionary in BatchOutputFormat format. The batch loader must provide SHUFFLED batches.
    task_type : str
        The task type of the data. can be one of the following: 'classification', 'semantic_segmentation',
        'object_detection', 'other'. For 'other', only image related checks (such as ImagePropertyOutliers) will be run.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    dataset_name: str, optional
        Name of the dataset to use in the displays instead of "Train" or "Test".
    shuffle_batch_loader: bool, default=True
        If True and the batch loader is of known type that can be shuffled, it will be shuffled.
    """

    def __init__(
            self,
            batch_loader,
            task_type: Literal['classification', 'object_detection', 'semantic_segmentation', 'other'],
            label_map: t.Optional[t.Dict[int, str]] = None,
            dataset_name: t.Optional[str] = None,
            shuffle_batch_loader: bool = True
    ):
        if not hasattr(batch_loader, '__iter__'):
            raise DeepchecksValueError(r'Batch loader must be an iterable which loads batches of data in deepcheck\'s'
                                       'required format, see link for additional information ')
        self._batch_loader = shuffle_loader(batch_loader) if shuffle_batch_loader else batch_loader

        if task_type not in TaskType.values():
            raise ValueError(f'Invalid task type: {task_type}, must be one of the following: {TaskType.values()}')
        self._task_type = TaskType(task_type)

        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError('label_map must be a dictionary')
        self._label_map = label_map
        self.name = dataset_name

        # indicator will be set to true in 'validate' method if the user implements the relevant formatters
        self._has_images, self._has_labels, self._has_predictions = False, False, False
        self._has_additional_data, self._has_embeddings, self._has_image_identifiers = False, False, False
        self.validate()
        self.init_cache()

    def init_cache(self):
        """Initialize the cache."""
        self._num_images_cached = 0
        # dict of class_id to number of images observed with label (num_label) and prediction (num_pred)
        self._observed_classes = defaultdict()

    def update_cache(self, batch_size, numpy_labels, numpy_predictions):
        """Update cache based on newly arrived batch."""
        self._num_images_cached += batch_size
        if numpy_labels is not None:
            for class_id, num_observed in get_class_ids_from_numpy_labels(numpy_labels, self._task_type).items():
                if self._label_map is not None and class_id not in self._label_map:
                    raise DeepchecksValueError(f'Class id {class_id} is not in the provided label map or out of bounds '
                                               f'for the given probability vector')
                if class_id not in self._observed_classes:
                    self._observed_classes[class_id] = {'num_label': 0, 'num_pred': 0}
                self._observed_classes[class_id]['num_label'] += num_observed

        if numpy_predictions is not None:
            for class_id, num_observed in get_class_ids_from_numpy_preds(numpy_predictions, self._task_type).items():
                if class_id not in self._observed_classes:
                    self._observed_classes[class_id] = {'num_label': 0, 'num_pred': 0}
                self._observed_classes[class_id]['num_pred'] += num_observed

    def validate(self):
        """Validate the VisionData functionalities implemented by the user and set which formatters were implemented."""
        batch: BatchOutputFormat = next(iter(self._batch_loader))
        allowed_keys = {'images', 'labels', 'predictions', 'additional_data', 'embeddings', 'image_identifiers'}
        if not isinstance(batch, dict) or not all(key in allowed_keys for key in batch.keys()):
            raise ValidationError('Batch loader batch output must be a dictionary containing a subset of the '
                                  f'following keys: {allowed_keys}.')
        length_dict = defaultdict()

        images = batch.get('images')
        if images is not None:
            self._has_images = True
            validate_images_format(images)
            length_dict['images'] = len(images)

        labels = batch.get('labels')
        if labels is not None:
            self._has_labels = True
            validate_labels_format(labels, self._task_type)
            length_dict['labels'] = len(labels)

        predictions = batch.get('predictions')
        if predictions is not None:
            self._has_predictions = True
            validate_predictions_format(predictions, self._task_type)
            if self._task_type == TaskType.CLASSIFICATION:
                if self._label_map is not None and len(predictions[0]) != len(self._label_map):
                    raise ValidationError('Number of entries in proba does not match number of classes in label_map')
                if self._label_map is None:
                    self._label_map = {i: str(i) for i in range(len(predictions[0]))}
            length_dict['predictions'] = len(predictions)

        additional_data = batch.get('additional_data')
        if additional_data is not None:
            self._has_additional_data = True
            validate_additional_data_format(additional_data)
            length_dict['additional_data'] = len(additional_data)

        embeddings = batch.get('embeddings')
        if embeddings is not None:
            self._has_embeddings = True
            validate_embeddings_format(embeddings)
            length_dict['embeddings'] = len(embeddings)

        if len(length_dict) == 0:  # TODO: use doc link once docs are available
            raise ValidationError('No data formatters were implemented, at least one of methods described in '
                                  'https://docs.deepchecks.com/stable/user-guide/vision/data-classes/VisionData.html'
                                  'must be implemented.')

        image_identifiers = batch.get('image_identifiers')
        if image_identifiers is not None:
            self._has_image_identifiers = True
            validate_image_identifiers_format(image_identifiers)
            length_dict['image_identifiers'] = len(image_identifiers)

        if len(set(length_dict.values())) > 1:
            raise ValidationError('All formatter functions must return sequences of the same length. '
                                  f'The following lengths were found: {length_dict}')

    @property
    def has_images(self) -> bool:
        """Return True if the batch loader contains images."""
        return self._has_images  # TODO: check also image path!

    @property
    def has_labels(self) -> bool:
        """Return True if the batch loader contains labels."""
        return self._has_labels

    @property
    def has_predictions(self) -> bool:
        """Return True if the batch loader contains predictions."""
        return self._has_predictions

    @property
    def has_embeddings(self) -> bool:
        """Return True if the batch loader contains embeddings."""
        return self._has_embeddings

    @property
    def has_additional_data(self) -> bool:
        """Return True if the batch loader contains additional_data."""
        return self._has_additional_data

    @property
    def has_image_identifiers(self) -> bool:
        """Return True if the batch loader contains image identifiers."""
        return self._has_image_identifiers

    @property
    def task_type(self) -> TaskType:
        """Return True if the batch loader contains labels."""
        return self._task_type

    @property
    def batch_loader(self):
        """Return the batch loader used be the vision data."""
        return self._batch_loader

    @property
    def number_of_images_cached(self) -> int:
        """Return True if the number of images processed and whose statistics were cached."""
        return self._num_images_cached

    @property
    def num_classes(self) -> int:
        """Return a number of possible classes based on model proba, label map or observed classes."""
        if self._label_map is not None:
            return len(self._label_map)
        else:
            return len(self._observed_classes)

    def get_observed_classes(self, use_class_names: bool = True) -> t.List[str]:
        """Return a dictionary of observed classes either as class ids or as the class names."""
        if use_class_names:
            return [self.label_id_to_name(x) for x in self._observed_classes.keys()]
        else:
            return list(self._observed_classes.keys())

    def get_cache(self, use_class_names: bool = True) -> t.Dict[str, t.Any]:
        """Return a dictionary of stored cache."""
        num_labels_per_class = {}
        num_preds_per_class = {}
        for key, value in self._observed_classes.items():
            key_name = self.label_id_to_name(key) if use_class_names else key
            if 'num_label' in value:
                num_labels_per_class[key_name] = value['num_label']
            if 'num_pred' in value:
                num_preds_per_class[key_name] = value['num_pred']

        return {'images_cached': self._num_images_cached, 'labels': num_labels_per_class,
                'predictions': num_preds_per_class}

    def label_id_to_name(self, class_id: int) -> str:
        """Return the name of the class with the given id."""
        class_id = int(class_id)
        if self._label_map is None:
            return str(class_id)
        elif class_id not in self._label_map:
            return str(class_id)
        else:
            return self._label_map[class_id]

    def copy(self, reshuffle_batch_loader: bool = False, batch_loader=None) -> VD:
        """Create new copy of the vision data object with clean cache.

        Parameters
        ----------
        reshuffle_batch_loader: bool, default=False
            If True and the batch loader is of known type that can be shuffled, it will be shuffled.
        batch_loader:
            If not None, the batch loader of the new object will be set to this value.
        Returns
        -------
        VisionData
            A copy of the vision data object with clean cache.
        """
        cls = type(self)
        batch_loader = batch_loader if batch_loader is not None else self._batch_loader
        return cls(batch_loader=batch_loader, task_type=self._task_type.value, label_map=self._label_map,
                   dataset_name=self.name, shuffle_batch_loader=reshuffle_batch_loader)

    def __iter__(self):
        """Return an iterator over the batch loader."""
        return iter(self._batch_loader)

    def __len__(self):
        """Return the number of batches in the batch loader if it is known, otherwise returns None."""
        return len(self._batch_loader) if hasattr(self._batch_loader, '__len__') else None
