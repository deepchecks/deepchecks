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
from typing import Tuple, Iterable, Any, TypeVar, Callable, cast

import torch

from deepchecks.core import DatasetKind


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

    @property
    def predictions(self):
        """Return predictions for the batch, formatted in deepchecks format."""
        if self._predictions is None:
            dataset = self._context.get_data_by_kind(self._dataset_kind)
            # Calling model will raise error if model was not given
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
            self._images = dataset.batch_to_images(self._batch)
        return self._images

    def __getitem__(self, index: int):
        """Return batch item by index."""
        return self._batch[index]

    def __len__(self):
        """Return length of batch."""
        return len(self._batch)

    def get_index_in_dataset(self, index: int) -> int:
        """For given index in this batch returns the real index in the underlying dataset object. Can be used to \
        later get samples for display."""
        dataset = self._context.get_data_by_kind(self._dataset_kind)
        return dataset.batch_index_to_dataset_index(self.batch_start_index + index)


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
