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
from abc import abstractmethod
import logging
from typing import List, Optional, Dict, Union
import numpy as np

import torch
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.vision_data import VisionData

logger = logging.getLogger('deepchecks')


class ClassificationData(VisionData):

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transforms'):

        super().__init__(data_loader, num_classes, label_map,
                         random_seed, transform_field)

        self._task_type = TaskType.CLASSIFICATION
        try:
            self._validate_label(next(iter(self._data_loader)))
            self._has_label = True
        except DeepchecksValueError:
            logger.warning('batch_to_labels() was not implemented, some checks will not run')
            self._has_label = False

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        """Infer on batch.
        Examples
        --------
        >>> return batch[1]
        """
        raise DeepchecksValueError(
            "batch_to_labels() must be implemented in a subclass"
        )

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        """Infer on batch.
        Examples
        --------
        >>> return model.to(device)(batch[0].to(device))
        """
        raise DeepchecksValueError(
            "infer_on_batch() must be implemented in a subclass"
        )

    def _validate_label(self, batch):
        """
        Validate the label.

        Parameters
        ----------
        batch
        """
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires classification label to be a torch.Tensor or numpy array')
        label_shape = labels.shape
        if len(label_shape) != 1:
            raise DeepchecksValueError('Check requires classification label to be a 1D tensor')

    def validate_prediction(self, model, device, n_classes: int = None, eps: float = 1e-3):
        """
        Validate the prediction.

        Parameters
        ----------
        batch : t.Any
            Batch as outputed from DataLoader
        model: t.Any
            Model to run on batch
        device: str
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
        batch_predictions = self.infer_on_batch(next(iter(self._data_loader)), model, device)
        if not isinstance(batch_predictions, (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires classification predictions to be a torch.Tensor or numpy '
                                       'array')
        pred_shape = batch_predictions.shape
        if len(pred_shape) != 2:
            raise DeepchecksValueError('Check requires classification predictions to be a 2D tensor')
        if n_classes and pred_shape[1] != n_classes:
            raise DeepchecksValueError(f'Check requires classification predictions to have {n_classes} columns')
        if any(abs(batch_predictions.sum(axis=1) - 1) > eps):
            raise DeepchecksValueError('Check requires classification} predictions to be a probability distribution and'
                                       ' sum to 1 for each row')
