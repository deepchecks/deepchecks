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

from deepchecks.core.errors import ValidationError
from deepchecks.utils.logger import get_logger
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
    dynamic_loader :
        A dynamic_loader which load a batch of data in an iterative manner. Dynamic loader batch output must be a
        dictionary in BatchOutputFormat format. The dynamic loader must provide SHUFFLED batches.
    task_type : str
        The task type of the data. can be one of the following: 'classification', 'semantic_segmentation',
        'object_detection', 'other'. For 'other', only image related checks (such as ImagePropertyOutliers) will be run.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    dataset_name: str, optional
        Name of the dataset to use in the displays instead of "Train" or "Test".
    shuffle_dynamic_loader: bool, default=True
        If True and the dynamic loader is of known type that can be shuffled, it will be shuffled.
    """

    def __init__(
            self,
            dynamic_loader,
            task_type: str,
            label_map: t.Optional[t.Dict[int, str]] = None,
            dataset_name: t.Optional[str] = None,
            shuffle_dynamic_loader: bool = True
    ):
        self._dynamic_loader = shuffle_loader(dynamic_loader) if shuffle_dynamic_loader else dynamic_loader

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
        self._num_images_seen = 0
        # dict of class_id to number of images observed with label (num_label) and prediction (num_pred)
        self._observed_classes = defaultdict()

    def update_cache(self, batch_size, numpy_labels, numpy_predictions):
        """update cache based on newly arrived batch."""
        self._num_images_seen += batch_size
        if numpy_labels is not None:
            for class_id, num_observed in get_class_ids_from_numpy_labels(numpy_labels, self._task_type).items():
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
        batch: BatchOutputFormat = next(iter(self._dynamic_loader))
        allowed_keys = {'images', 'labels', 'predictions', 'additional_data', 'embeddings', 'image_identifiers'}
        if not isinstance(batch, dict) or not all(key in allowed_keys for key in batch.keys()):
            raise ValidationError('Dynamic loader batch output must be a dictionary containing a subset of the '
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

        if len(length_dict) == 0:  # TODO: use doclink once docs are available
            raise ValidationError('No data formatters were implemented, at least one of methods described in '
                                  'https://docs.deepchecks.com/stable/user-guide/vision/data-classes/VisionData.html'
                                  'must be implemented.')

        image_identifiers = batch.get('image_identifiers')
        if image_identifiers is not None:
            self._has_image_identifiers = True
            validate_image_identifiers_format(image_identifiers)
            length_dict['image_identifiers'] = len(image_identifiers)

        if not all(ele == list(length_dict.values())[0] for ele in length_dict.values()):
            raise ValidationError('All formatter functions must return sequences of the same length. '
                                  f'The following lengths were found: {length_dict}')

    @property
    def has_images(self) -> bool:
        """Return True if the dynamic loader contains images."""
        return self._has_images  # TODO: check also image path!

    @property
    def has_labels(self) -> bool:
        """Return True if the dynamic loader contains labels."""
        return self._has_labels

    @property
    def has_predictions(self) -> bool:
        """Return True if the dynamic loader contains predictions."""
        return self._has_predictions

    @property
    def has_embeddings(self) -> bool:
        """Return True if the dynamic loader contains embeddings."""
        return self._has_embeddings

    @property
    def has_additional_data(self) -> bool:
        """Return True if the dynamic loader contains additional_data."""
        return self._has_embeddings

    @property
    def has_image_identifiers(self) -> bool:
        """Return True if the dynamic loader contains image identifiers."""
        return self._has_image_identifiers

    @property
    def task_type(self) -> TaskType:
        """Return True if the dynamic loader contains labels."""
        return self._task_type

    @property
    def dynamic_loader(self):
        """Return the dynamic loader used be the vision data."""
        return self._dynamic_loader

    @property
    def number_of_images_cached(self) -> int:
        """Return True if the number of images processed and whose statistics were cached."""
        return self._num_images_seen

    @property
    def observed_classes(self) -> t.List[str]:
        """Return a dictionary of observed classes as true label name."""
        return [self.label_id_to_name(x) for x in self._observed_classes.keys()]

    def get_cache(self) -> t.Dict[str, t.Any]:
        """Return a dictionary of stored cache."""
        num_labels_per_class_id = {}
        num_preds_per_class_id = {}
        for key, value in self._observed_classes.items():
            if 'num_label' in value:
                num_labels_per_class_id[self.label_id_to_name(key)] = value['num_label']
            if 'num_pred' in value:
                num_preds_per_class_id[self.label_id_to_name(key)] = value['num_pred']

        return {'num_images_seen': self._num_images_seen, 'labels': num_labels_per_class_id,
                'predictions': num_preds_per_class_id}

    def label_id_to_name(self, class_id: int) -> str:
        """Return the name of the class with the given id."""
        class_id = int(class_id)
        if self._label_map is None:
            return str(class_id)
        elif class_id not in self._label_map:
            get_logger().warning('Class id %s is not in the label map. Add it to map '
                                 'in order to show the class name instead of id', class_id)
            return str(class_id)
        else:
            return self._label_map[class_id]

    def copy(self, reshuffle_dynamic_loader: bool = False, dynamic_loader=None) -> VD:
        """Create new copy of the vision data object with clean cache.

        Parameters
        ----------
        reshuffle_dynamic_loader: bool, default=False
            If True and the dynamic loader is of known type that can be shuffled, it will be shuffled.
        dynamic_loader:
            If not None, the dynamic loader of the new object will be set to this value.
        Returns
        -------
        VisionData
            A copy of the vision data object with clean cache.
        """
        cls = type(self)
        dynamic_loader = dynamic_loader if dynamic_loader is not None else self._dynamic_loader
        return cls(dynamic_loader=dynamic_loader, task_type=self._task_type.value, label_map=self._label_map,
                   dataset_name=self.name, reshuffle_dynamic_loader=reshuffle_dynamic_loader)

    def __iter__(self):
        """Return an iterator over the dynamic loader."""
        return iter(self._dynamic_loader)

    def __len__(self):
        """Return the number of batches in the dynamic loader if it is known, otherwise returns None."""
        return len(self._dynamic_loader) if hasattr(self._dynamic_loader, '__len__') else None
