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
import sys
import typing as t
from collections import defaultdict

import numpy as np
from IPython.core.display import display
from ipywidgets import HTML
from typing_extensions import Literal

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.utils.ipython import is_notebook, is_sphinx
from deepchecks.vision.utils.detection_formatters import DEFAULT_PREDICTION_FORMAT
from deepchecks.vision.utils.image_functions import draw_bboxes, draw_masks, prepare_thumbnail, random_color_dict
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
from deepchecks.vision.vision_data.format_validators import (validate_additional_data_format,
                                                             validate_embeddings_format,
                                                             validate_image_identifiers_format, validate_images_format,
                                                             validate_labels_format, validate_predictions_format)
from deepchecks.vision.vision_data.utils import (BatchOutputFormat, LabelMap, get_class_ids_from_numpy_labels,
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
    reshuffle_data: bool, default=True
        If True we will attempt to shuffle the batch loader. Only set this to False if the data is already shuffled.
    """

    def __init__(
            self,
            batch_loader,
            task_type: Literal['classification', 'object_detection', 'semantic_segmentation', 'other'],
            label_map: t.Optional[t.Dict[int, str]] = None,
            dataset_name: t.Optional[str] = None,
            reshuffle_data: bool = True
    ):
        if not hasattr(batch_loader, '__iter__'):
            # TODO: add link to documentation
            raise DeepchecksValueError(r'Batch loader must be an iterable which loads batches of data in deepcheck\'s'
                                       'required format, see link for additional information ')
        self._batch_loader = shuffle_loader(batch_loader) if reshuffle_data else batch_loader

        if task_type not in TaskType.values():
            raise ValueError(f'Invalid task type: {task_type}, must be one of the following: {TaskType.values()}')
        self._task_type = TaskType(task_type)

        if label_map is not None and not isinstance(label_map, dict):
            raise ValueError('label_map must be a dictionary')
        self.label_map = LabelMap(label_map)
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
        if numpy_labels is not None and self.task_type != TaskType.OTHER:
            for class_id, num_observed in get_class_ids_from_numpy_labels(numpy_labels, self._task_type).items():
                if self.label_map and class_id not in self.label_map:
                    raise DeepchecksValueError(f'Class id {class_id} is not in the provided label map or out of bounds '
                                               f'for the given probability vector')
                if class_id not in self._observed_classes:
                    self._observed_classes[class_id] = {'num_label': 0, 'num_pred': 0}
                self._observed_classes[class_id]['num_label'] += num_observed

        if numpy_predictions is not None and self.task_type != TaskType.OTHER:
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
                if self.label_map and len(predictions[0]) != len(self.label_map):
                    raise ValidationError('Number of entries in proba does not match number of classes in label_map')
                if not self.label_map:
                    self.label_map = LabelMap({i: str(i) for i in range(len(predictions[0]))})
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
        if self.label_map:
            return len(self.label_map)
        else:
            return len(self._observed_classes)

    def get_observed_classes(self, use_class_names: bool = True) -> t.List[str]:
        """Return a dictionary of observed classes either as class ids or as the class names."""
        if use_class_names:
            return [self.label_map[x] for x in self._observed_classes.keys()]
        else:
            return list(self._observed_classes.keys())

    def get_cache(self, use_class_names: bool = True) -> t.Dict[str, t.Any]:
        """Return a dictionary of stored cache."""
        num_labels_per_class = {}
        num_preds_per_class = {}
        for key, value in self._observed_classes.items():
            key_name = self.label_map[key] if use_class_names else key
            if 'num_label' in value:
                num_labels_per_class[key_name] = value['num_label']
            if 'num_pred' in value:
                num_preds_per_class[key_name] = value['num_pred']

        return {'images_cached': self._num_images_cached, 'labels': num_labels_per_class,
                'predictions': num_preds_per_class}

    def copy(self, reshuffle_data: bool = False, batch_loader=None) -> VD:
        """Create new copy of the vision data object with clean cache.

        Parameters
        ----------
        reshuffle_data: bool, default=False
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
        return cls(batch_loader=batch_loader, task_type=self._task_type.value, label_map=self.label_map,
                   dataset_name=self.name, reshuffle_data=reshuffle_data)

    def __iter__(self):
        """Return an iterator over the batch loader."""
        return iter(self._batch_loader)

    def __len__(self):
        """Return the number of batches in the batch loader if it is known, otherwise returns None."""
        return len(self._batch_loader) if hasattr(self._batch_loader, '__len__') else None

    def head(self, num_images_to_display: int = 5):
        """Show data from a single batch of this VisionData. Works only inside a notebook.

        Parameters
        ----------
        num_images_to_display: int, default = 5
            Number of images to show. Does not show more images than the size of single batch
        """
        if not (is_notebook() or is_sphinx()):
            print('head function is supported only inside a notebook', file=sys.stderr)
            return
        if not isinstance(num_images_to_display, int):
            print('num_images_to_display must be an integer')
            return
        if num_images_to_display < 1:
            print('num_images_to_display can\'t be smaller than 1', file=sys.stderr)
            return

        image_size = (300, 300)
        images = []
        headers_row = []
        rows = [[] for _ in range(num_images_to_display)]
        color_dict = None
        batch = BatchWrapper(next(iter(self._batch_loader)), self.task_type, self.number_of_images_cached)

        if self.task_type == TaskType.SEMANTIC_SEGMENTATION:
            # Creating a colors dict to be shared for all images
            num_classes = 0
            if self.has_predictions:
                num_classes = batch.numpy_predictions[0].shape[0]
            elif self.has_labels:
                num_classes = max(np.max(label) for label in batch.numpy_labels[:num_images_to_display])

            color_dict = random_color_dict(num_classes)

        if self.has_image_identifiers:
            headers_row.append('<h4>Identifier</h4>')
            for index, image_id in enumerate(batch.numpy_image_identifiers[:num_images_to_display]):
                rows[index].append(f'<p style="overflow-wrap: anywhere;font-size:2em;">{image_id}</p>')

        if self.has_images:
            headers_row.append('<h4>Images</h4>')
            images = batch.numpy_images[:num_images_to_display]
            for index, image in enumerate(images):
                rows[index].append(prepare_thumbnail(image, size=image_size))

        if self.has_labels:
            headers_row.append('<h4>Labels</h4>')
            labels = batch.numpy_labels[:num_images_to_display]
            for index, label in enumerate(labels):
                if self.task_type == TaskType.OBJECT_DETECTION:
                    label_image = draw_bboxes(images[index], label, self.label_map, copy_image=False, border_width=5)
                    rows[index].append(prepare_thumbnail(label_image, size=image_size))
                elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
                    label_image = draw_masks(images[index], label, copy_image=False, color=color_dict)
                    rows[index].append(prepare_thumbnail(label_image, size=image_size))
                else:
                    rows[index].append(f'<p style="overflow-wrap: anywhere;font-size:2em;">'
                                       f'{self.label_map[label]}</p>')

        if self.has_predictions:
            headers_row.append('<h4>Predictions</h4>')
            predictions = batch.numpy_predictions[:num_images_to_display]
            for index, prediction in enumerate(predictions):
                if self.task_type == TaskType.OBJECT_DETECTION:
                    pred_image = draw_bboxes(images[index], prediction, self.label_map, copy_image=False, color='blue',
                                             border_width=5, bbox_notation=DEFAULT_PREDICTION_FORMAT)
                    rows[index].append(prepare_thumbnail(pred_image, size=image_size))
                elif self.task_type == TaskType.SEMANTIC_SEGMENTATION:
                    # Convert C,H,W to single mask with all classes of shape H,W
                    prediction = np.argmax(prediction, axis=0)
                    pred_image = draw_masks(images[index], prediction, copy_image=False, color=color_dict)
                    rows[index].append(prepare_thumbnail(pred_image, size=image_size))
                else:
                    prediction = np.argmax(prediction)
                    rows[index].append(f'<p style="overflow-wrap: anywhere;font-size:2em;">'
                                       f'{self.label_map[prediction]}</p>')

        html = '<div style="display:flex; flex-direction: column; gap: 10px;">'

        for row in [headers_row] + rows:
            inner = [f'<div style="place-self: center;">{i}</div>' for i in row]
            html += f"""
                <div style="display: grid; grid-auto-columns: minmax(0, 1fr); grid-auto-flow: column; gap:10px;">
                    {"".join(inner)}
                </div>
            """
        html += '</div>'

        if is_notebook():
            display(HTML(html))
        else:
            class TempSphinx:
                def _repr_html_(self):
                    return html

            return TempSphinx()
