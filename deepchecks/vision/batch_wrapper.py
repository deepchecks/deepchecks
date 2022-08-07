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
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple, TypeVar, cast

import torch

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.vision_properties import (PropertiesInputType, calc_vision_properties,
                                                       static_prop_to_cache_format, validate_properties)

if TYPE_CHECKING:
    from deepchecks.vision.context import Context


__all__ = ['Batch']


class Batch:
    """Represents dataset batch returned by the dataloader during iteration."""

    def __init__(
        self,
        batch: Tuple[Iterable[Any], Iterable[Any]],
        context: 'Context',  # noqa
        dataset_kind: DatasetKind,
        batch_index: int
    ):
        self._context = context
        self._dataset_kind = dataset_kind
        self.batch_index = batch_index
        self._batch = apply_to_tensor(batch, lambda it: it.to(self._context.device))
        self._labels = None
        self._predictions = None
        self._images = None
        self._vision_properties_cache = dict.fromkeys([p.value for p in PropertiesInputType])

    @property
    def labels(self):
        """Return labels for the batch, formatted in deepchecks format."""
        if self._labels is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            dataset.assert_labels_valid()
            self._labels = dataset.batch_to_labels(self._batch)
        return self._labels

    def _do_static_pred(self):
        preds = self._context.static_predictions[self._dataset_kind]
        dataset = self._context.get_data_by_kind(self._dataset_kind)
        indexes = list(dataset.data_loader.batch_sampler)[self.batch_index]
        preds = itemgetter(*indexes)(preds)
        if dataset.task_type == TaskType.CLASSIFICATION:
            return torch.stack(preds)
        return preds

    @property
    def predictions(self):
        """Return predictions for the batch, formatted in deepchecks format."""
        if self._predictions is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            if self._context.static_predictions is not None:
                self._context.assert_predictions_valid(self._dataset_kind)
                self._predictions = self._do_static_pred()
            else:
                # Calling model will raise error if model was not given
                # (assert_predictions_valid doesn't raise an error if no model was given)
                model = self._context.model
                self._context.assert_predictions_valid(self._dataset_kind)
                self._predictions = dataset.infer_on_batch(self._batch, model, self._context.device)
        return self._predictions

    @property
    def images(self):
        """Return images for the batch, formatted in deepchecks format."""
        if self._images is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            dataset.assert_images_valid()
            self._images = [image.astype('uint8') for image in dataset.batch_to_images(self._batch)]
        return self._images

    def __getitem__(self, index: int):
        """Return batch item by index."""
        return self._batch[index]

    def __len__(self):
        """Return length of batch."""
        dataset = self._context.get_data_by_kind(self._dataset_kind)
        return len(list(dataset.data_loader.batch_sampler)[self.batch_index])

    def _do_static_prop(self):
        """Get a batch of static properties and transform it to the cache format."""
        props = self._context.static_properties[self._dataset_kind]
        dataset = self._context.get_data_by_kind(self._dataset_kind)
        indexes = list(dataset.data_loader.batch_sampler)[self.batch_index]
        props_to_cache = static_prop_to_cache_format(dict(zip(indexes, props)))
        return props_to_cache

    def vision_properties(self, raw_data: List, properties_list: List[Dict], input_type: PropertiesInputType):
        """Calculate and cache the properties for the batch according to the property input type."""
        properties_list = validate_properties(properties_list)
        # if there are no cached properties at all, calculate all the properties on the list,
        # else calculate only those that were not yet calculated.
        if self._vision_properties_cache[input_type.value] is None:
            if self._context.static_properties is not None:
                self._vision_properties_cache = self._do_static_prop()
            else:
                self._vision_properties_cache[input_type.value] = calc_vision_properties(raw_data, properties_list)
            result_dict = self._vision_properties_cache[input_type.value]
        else:
            properties_to_calc = [p for p in properties_list if p['name'] not in
                                  self._vision_properties_cache[input_type.value].keys()]
            self._vision_properties_cache[input_type.value].update(
                                                                calc_vision_properties(raw_data, properties_to_calc))
            property_names_to_return = [p['name'] for p in properties_list]
            result_dict = {k: self._vision_properties_cache[input_type.value][k] for k in property_names_to_return}
        # validate properties length
        static_prop_lens = [len(pr) for pr in result_dict.values()]
        if any([(p != len(raw_data) and p != 1) for p in static_prop_lens]):  # pylint: disable=use-a-generator
            raise DeepchecksProcessError('The properties should have the same length as the raw data')
        return result_dict


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
