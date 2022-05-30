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
from typing import TYPE_CHECKING, Any, Callable, Iterable, Tuple, TypeVar, cast

import torch

from deepchecks.core import DatasetKind

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
        batch_start_index: int
    ):
        self._context = context
        self._dataset_kind = dataset_kind
        self.batch_start_index = batch_start_index
        self._batch = apply_to_tensor(batch, lambda it: it.to(self._context.device))
        self._labels = None
        self._predictions = None
        self._images = None

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
        indexes = [dataset.to_dataset_index(self.batch_start_index + index)[0]
                   for index in range(len(self))]
        if isinstance(preds, torch.Tensor):
            return preds[indexes]
        return itemgetter(*indexes)(preds)

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
        dataset_len = dataset.num_samples
        dataloader_len = len(dataset.data_loader)
        max_len = int(dataset_len / dataloader_len + 0.5)
        if self.batch_start_index + max_len > dataset_len:  # last batch
            return dataset_len - self.batch_start_index
        return max_len

    def get_index_in_dataset(self, index: int) -> int:
        """For given index in this batch returns the real index in the underlying dataset object. Can be used to \
        later get samples for display."""
        dataset = self._context.get_data_by_kind(self._dataset_kind)
        return dataset.to_dataset_index(self.batch_start_index + index)[0]


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
