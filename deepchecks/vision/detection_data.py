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

class DetectionData(VisionData):
    """
    DetectionData is an abstract class that defines the interface for
    object detection tasks.

    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 sample_size: int = 1000,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transforms'):

        super().__init__(data_loader, num_classes, label_map, sample_size,
                         random_seed, transform_field)
        self.task_type = TaskType.OBJECT_DETECTION
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

        Returns
        -------
        Optional[str]
            None if the label is valid, otherwise a string containing the error message.

        """
        labels = self.batch_to_labels(batch)
        if not isinstance(labels, list):
            raise DeepchecksValueError('Check requires object detection label to be a list with an entry for each '
                                       'sample')
        if len(labels) == 0:
            raise DeepchecksValueError('Check requires object detection label to be a non-empty list')
        if not isinstance(labels[0], (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires object detection label to be a list of torch.Tensor or numpy '
                                       'array')
        if len(labels[0].shape) != 2:
            raise DeepchecksValueError('Check requires object detection label to be a list of 2D tensors')
        if labels[0].shape[1] != 5:
            raise DeepchecksValueError('Check requires object detection label to be a list of 2D tensors, when '
                                       'each row has 5 columns: [class_id, x, y, width, height]')

    def validate_prediction(self, batch, model, device):
        """
        Validate the prediction.

        Parameters
        ----------
        batch : t.Any
            Batch from DataLoader
        model : t.Any
        device : torch.Device
        """
        batch_predictions = self.infer_on_batch(batch, model, device)
        if not isinstance(batch_predictions, list):
            raise DeepchecksValueError('Check requires detection predictions to be a list with an entry for each'
                                       ' sample')
        if len(batch_predictions) == 0:
            raise DeepchecksValueError('Check requires detection predictions to be a non-empty list')
        if not isinstance(batch_predictions[0], (torch.Tensor, np.ndarray)):
            raise DeepchecksValueError('Check requires detection predictions to be a list of torch.Tensor or'
                                       ' numpy array')
        if len(batch_predictions[0].shape) != 2:
            raise DeepchecksValueError('Check requires detection predictions to be a list of 2D tensors')
        if batch_predictions[0].shape[1] != 6:
            raise DeepchecksValueError('Check requires detection predictions to be a list of 2D tensors, when '
                                       'each row has 6 columns: [x, y, width, height, class_probability, class_id]')
