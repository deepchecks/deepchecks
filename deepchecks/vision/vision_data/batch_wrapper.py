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
"""Contains code for BatchWrapper."""
from typing import Dict, List, Optional, Union

import numpy as np

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.image_properties import calc_default_image_properties, default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType, calc_vision_properties, validate_properties
from deepchecks.vision.vision_data.utils import BatchOutputFormat, TaskType, sequence_to_numpy

__all__ = ['BatchWrapper']


class BatchWrapper:
    """Represents dataset batch returned by the dataloader during iteration."""

    def __init__(self, batch: BatchOutputFormat, task_type: TaskType, images_seen_num: int):
        self._task_type = task_type
        self._batch = batch
        self._labels, self._predictions, self._images = None, None, None
        self._embeddings, self._additional_data, = None, None
        self._image_identifiers = batch.get('image_identifiers')
        # if there are no image identifiers, use the number of the image in loading process as identifier
        if self._image_identifiers is None:
            self._image_identifiers = np.asarray(range(images_seen_num, images_seen_num + len(self)), dtype='str')

        self._vision_properties_cache = dict.fromkeys(PropertiesInputType)

    def _get_relevant_data_for_properties(self, input_type: PropertiesInputType):
        result = []
        if input_type == PropertiesInputType.PARTIAL_IMAGES:
            for img, bboxes_in_img in zip(self.numpy_images, self.numpy_labels):
                if bboxes_in_img is None:
                    continue
                result = result + [crop_image(img, *bbox[1:]) for bbox in bboxes_in_img]
        elif input_type == PropertiesInputType.IMAGES:
            result = self.numpy_images
        elif input_type == PropertiesInputType.LABELS:
            result = self.numpy_labels
        elif input_type == PropertiesInputType.PREDICTIONS:
            result = self.numpy_predictions
        return result

    def vision_properties(self, properties_list: Optional[List[Dict]], input_type: PropertiesInputType):
        """Calculate and cache the properties for the batch according to the property input type.

        Parameters
        ----------
        properties_list: Optional[List[Dict]]
            List of properties to calculate. If None, default properties will be calculated.
        input_type: PropertiesInputType
            The input type of the properties.

        Returns
        -------
        Dict[str, Any]
            Dictionary of the properties name to list of property values per data element.
        """
        if self._vision_properties_cache[input_type] is None:
            self._vision_properties_cache[input_type] = {}
        keys_in_cache = self._vision_properties_cache[input_type].keys()

        if properties_list is not None:
            properties_list = validate_properties(properties_list)
            requested_properties_names = [prop['name'] for prop in properties_list]
            properties_to_calc = [p for p in properties_list if p['name'] not in keys_in_cache]
            if len(properties_to_calc) > 0:
                data = self._get_relevant_data_for_properties(input_type)
                self._vision_properties_cache[input_type].update(calc_vision_properties(data, properties_to_calc))
        else:
            if input_type not in [PropertiesInputType.PARTIAL_IMAGES, PropertiesInputType.IMAGES]:
                # TODO: add support for quick default properties calculation for other input types
                raise DeepchecksProcessError(f'None was passed to properties calculation for input type {input_type}.')
            requested_properties_names = [prop['name'] for prop in default_image_properties]
            if any(x not in keys_in_cache for x in requested_properties_names):
                data = self._get_relevant_data_for_properties(input_type)
                self._vision_properties_cache[input_type].update(calc_default_image_properties(data))

        return {key: value for key, value in self._vision_properties_cache[input_type].items() if
                key in requested_properties_names}

    @property
    def original_labels(self):
        """Return labels for the batch, formatted in deepchecks format."""
        if self._labels is None:
            self._labels = self._batch.get('labels')
        return self._labels

    @property
    def numpy_labels(self) -> List[Union[np.ndarray, int]]:
        """Return labels for the batch in numpy format."""
        required_dim = 0 if self._task_type == TaskType.CLASSIFICATION else 2
        return sequence_to_numpy(self.original_labels, expected_ndim_per_object=required_dim)

    @property
    def original_predictions(self):
        """Return predictions for the batch, formatted in deepchecks format."""
        if self._predictions is None:
            self._predictions = self._batch.get('predictions')
        return self._predictions

    @property
    def numpy_predictions(self) -> List[np.ndarray]:
        """Return predictions for the batch in numpy format."""
        if self._task_type == TaskType.CLASSIFICATION:
            required_dim = 1
        elif self._task_type == TaskType.OBJECT_DETECTION:
            required_dim = 2
        elif self._task_type == TaskType.SEMANTIC_SEGMENTATION:
            required_dim = 3
        else:
            required_dim = None
        return sequence_to_numpy(self.original_predictions, expected_ndim_per_object=required_dim)

    @property
    def original_images(self):
        """Return images for the batch, formatted in deepchecks format."""
        if self._images is None:
            self._images = self._batch.get('images')
        return self._images

    @property
    def numpy_images(self) -> List[Union[np.ndarray]]:
        """Return images for the batch in numpy format."""
        return sequence_to_numpy(self.original_images, 'uint8', 3)

    @property
    def original_embeddings(self):
        """Return embedding for the batch, formatted in deepchecks format."""
        if self._embeddings is None:
            self._embeddings = self._batch.get('embeddings')
        return self._embeddings

    @property
    def numpy_embeddings(self) -> List[Union[np.ndarray]]:
        """Return embedding for the batch in numpy format."""
        return sequence_to_numpy(self.original_embeddings, 'float32')

    @property
    def original_additional_data(self):
        """Return additional data for the batch, formatted in deepchecks format."""
        if self._additional_data is None:
            self._additional_data = self._batch.get('additional_data')
        return self._additional_data

    @property
    def numpy_additional_data(self):
        """Return additional data for the batch in numpy format."""
        return sequence_to_numpy(self.original_additional_data)

    @property
    def original_image_identifiers(self):
        """Return image identifiers for the batch, formatted in deepchecks format."""
        return self._image_identifiers

    @property
    def numpy_image_identifiers(self) -> List[Union[str, int]]:
        """Return image identifiers for the batch in numpy format."""
        return sequence_to_numpy(self.original_image_identifiers, 'str', 0)

    def __len__(self):
        """Return length of batch."""
        data = self.numpy_images if self.numpy_images is not None else self.numpy_predictions if \
            self.numpy_predictions is not None else self.numpy_labels if self.numpy_labels is not None else \
            self.numpy_embeddings if self.numpy_embeddings is not None else self.numpy_additional_data
        return len(data)
