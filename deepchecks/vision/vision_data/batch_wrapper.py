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
from typing import Any, Callable, Dict, List, Sequence, TypeVar, Union, cast

import numpy as np
import torch

from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.vision_properties import PropertiesInputType, calc_vision_properties, validate_properties
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.vision_data.utils import BatchOutputFormat, sequence_to_numpy

__all__ = ['BatchWrapper']


class BatchWrapper:
    """Represents dataset batch returned by the dataloader during iteration."""

    def __init__(
            self,
            batch: BatchOutputFormat,
            vision_data: VisionData,
    ):
        self._vision_data = vision_data
        self._batch = batch
        self._labels, self._predictions, self._images = None, None, None
        self._embeddings, self._additional_data, = None, None
        self._image_identifiers = batch.get('image_identifiers')
        # if there are no image identifiers, use the number of  the image in loading process as identifier
        if self._image_identifiers is None:
            images_seen = self._vision_data.number_of_images_cached
            self._image_identifiers = np.asarray(range(images_seen, images_seen + self.__len__()), dtype='str')

        self._vision_properties_cache = dict.fromkeys(PropertiesInputType)

    def _get_cropped_images(self):
        imgs = []
        for img, labels in zip(self.images, self.labels):
            for label in labels:
                label = label.cpu().detach().numpy()
                bbox = label[1:]
                # make sure image is not out of bounds
                if round(bbox[2]) + min(round(bbox[0]), 0) <= 0 or round(bbox[3]) <= 0 + min(round(bbox[1]), 0):
                    continue
                imgs.append(crop_image(img, *bbox))
        return imgs

    def _get_relevant_data_for_properties(self, input_type: PropertiesInputType):
        if input_type == PropertiesInputType.PARTIAL_IMAGES:
            return self._get_cropped_images()
        if input_type == PropertiesInputType.IMAGES:
            return self._batch.get('images')
        if input_type == PropertiesInputType.LABELS:
            return self._batch.get('labels')
        if input_type == PropertiesInputType.PREDICTIONS:
            return self._batch.get('predictions')

    def vision_properties(self, properties_list: List[Dict], input_type: PropertiesInputType):
        """Calculate and cache the properties for the batch according to the property input type."""
        properties_list = validate_properties(properties_list)
        # if there are no cached properties at all, calculate all the properties on the list,
        # else calculate only those that were not yet calculated.
        if self._vision_properties_cache[input_type] is None:
            data = self._get_relevant_data_for_properties(input_type)
            self._vision_properties_cache[input_type] = calc_vision_properties(data, properties_list)
        else:
            properties_to_calc = [p for p in properties_list if p['name'] not in
                                  self._vision_properties_cache[input_type].keys()]
            if len(properties_to_calc) > 0:
                data = self._get_relevant_data_for_properties(input_type)
                self._vision_properties_cache[input_type].update(calc_vision_properties(data, properties_to_calc))
        return self._vision_properties_cache[input_type]

    @property
    def original_labels(self):
        """Return labels for the batch, formatted in deepchecks format."""
        if self._labels is None:
            self._labels = self._batch.get('labels')
        return self._labels

    @property
    def numpy_labels(self) -> Sequence[Union[np.ndarray, int]]:
        """Return labels for the batch in numpy format."""
        return sequence_to_numpy(self.original_labels)

    @property
    def original_predictions(self):
        """Return predictions for the batch, formatted in deepchecks format."""
        if self._predictions is None:
            self._predictions = self._batch.get('predictions')
        return self._predictions

    @property
    def numpy_predictions(self) -> Sequence[Union[np.ndarray]]:
        """Return predictions for the batch in numpy format."""
        return sequence_to_numpy(self.original_predictions)

    @property
    def original_images(self):
        """Return images for the batch, formatted in deepchecks format."""
        if self._images is None:
            self._images = self._batch.get('images')
        return self._images

    @property
    def numpy_images(self) -> Sequence[Union[np.ndarray]]:
        """Return images for the batch in numpy format."""
        return sequence_to_numpy(self.original_images, 'uint8')

    @property
    def original_embeddings(self):
        """Return embedding for the batch, formatted in deepchecks format."""
        if self._embeddings is None:
            self._embeddings = self._batch.get('embeddings')
        return self._embeddings

    @property
    def numpy_embeddings(self) -> Sequence[Union[np.ndarray]]:
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
    def numpy_image_identifiers(self) -> Sequence[Union[str, int]]:
        """Return image identifiers for the batch in numpy format."""
        return sequence_to_numpy(self.original_image_identifiers, 'str')

    def __len__(self):
        """Return length of batch."""
        data = self.numpy_images if self.numpy_images is not None else self.numpy_predictions \
            if self.numpy_predictions is not None else self.numpy_labels if self.numpy_labels is not None else \
            self.numpy_embeddings if self.numpy_embeddings is not None else self.numpy_additional_data
        return len(data)


# TODO: delete below

T = TypeVar('T')


def apply_to_tensor(
        x: T,
        fn: Callable[[torch.Tensor], torch.Tensor]
) -> Any:
    """Apply provided function to tensor instances recursivly."""
    if isinstance(x, torch.Tensor):
        return cast(T, fn(x))
    elif isinstance(x, (str, bytes, bytearray)):
        return x
    elif isinstance(x, (list, tuple, set)):
        return type(x)(apply_to_tensor(it, fn) for it in x)
    elif isinstance(x, dict):
        return type(x)((k, apply_to_tensor(v, fn)) for k, v in x.items())
    return x
