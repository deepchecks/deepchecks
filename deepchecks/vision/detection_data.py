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
from itertools import chain
import logging
from typing import List, Optional, Dict, Union
import numpy as np

import torch
from torch.utils.data import DataLoader

from deepchecks.core.errors import DeepchecksNotImplementedError, DeepchecksValueError, ValidationError
from deepchecks.vision.vision_data import VisionData, TaskType

logger = logging.getLogger('deepchecks')


class DetectionData(VisionData):
    """The ClassificationData class is used to load and preprocess data for a object detection task.

    It is a subclass of the VisionData class. The DetectionData class is containing additional data and general
    methods intended for easily accessing metadata relevant for validating a computer vision object detection ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. This is the data loader object that will be used to load the data.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 transform_field: Optional[str] = 'transforms'):

        super().__init__(data_loader, num_classes, label_map, transform_field)

        self._task_type = TaskType.OBJECT_DETECTION
        self._has_label = False

        try:
            self.validate_label(next(iter(self._data_loader)))
            self._has_label = True
        except DeepchecksNotImplementedError:
            logger.warning('batch_to_labels() was not implemented, some checks will not run')
        except ValidationError as ex:
            logger.warning('batch_to_labels() was not implemented correctly, '
                           'the validiation has failed with the error: %s', {str(ex)})

    @abstractmethod
    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        """Extract the labels from a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            The labels extracted from the batch. The labels should be a list of length N containing tensor of shape
            (B, 5) where N is the number of samples, B is the number of bounding boxes in the sample and each bounding
            box is represented by 5 values. See the notes for more info.

        Examples
        --------
        >>> import torch
        ...
        ...
        ... def batch_to_labels(self, batch):
        ...     # each bbox in the labels is (class_id, x, y, x, y). convert to (class_id, x, y, w, h)
        ...     return [torch.stack(
        ...            [torch.cat((bbox[0], bbox[1:3], bbox[4:] - bbox[1:3]), dim=0)
        ...                for bbox in image])
        ...             for image in batch[1]]

        Notes
        -----
        The accepted label format for is a a list of length N containing tensors of shape (B, 5), where N is the number
        of samples, B is the number of bounding boxes in the sample and each bounding box is represented by 5 values:
        (class_id, x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w
         and h are the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.
        """
        raise DeepchecksNotImplementedError('batch_to_labels() must be implemented in a subclass')

    @abstractmethod
    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        """Return the predictions of the model on a batch of data.

        Parameters
        ----------
        batch : torch.Tensor
            The batch of data.
        model : torch.nn.Module
            The model to use for inference.
        device : torch.device
            The device to use for inference.

        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            The predictions of the model on the batch. The predictions should be in a List of length N containing
            tensors of shape (B, 6), where N is the number of images, B is the number of bounding boxes detected in the
            sample and each bounding box is represented by 6 values. See the notes for more info.

        Examples
        --------
        >>> import torch
        ...
        ...
        ... def infer_on_batch(self, batch, model, device):
        ...     # Converts a yolo prediction batch to the accepted xywh format
        ...     return_list = []
        ...
        ...     predictions = model(batch[0])
        ...     # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        ...     for single_image_tensor in predictions.pred:
        ...         pred_modified = torch.clone(single_image_tensor)
        ...         pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]
        ...         pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]
        ...         return_list.append(pred_modified)
        ...
        ...     return return_list

        Notes
        -----
        The accepted prediction format is a list of length N containing tensors of shape (B, 6), where N is the number
        of images, B is the number of bounding boxes detected in the sample and each bounding box is represented by 6
        values: [x, y, w, h, confidence, class_id]. x and y are the coordinates (in pixels) of the upper left corner
        of the bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the
        confidence of the model and class_id is the class id.
        """
        raise DeepchecksNotImplementedError('infer_on_batch() must be implemented in a subclass')

    def get_classes(self, batch_labels: List[torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        def get_classes_from_single_label(tensor: torch.Tensor):
            return list(tensor[:, 0].tolist()) if len(tensor) > 0 else []

        return list(chain(*[get_classes_from_single_label(x) for x in batch_labels]))

    def validate_label(self, batch):
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
        if not isinstance(labels[0], torch.Tensor):
            raise DeepchecksValueError('Check requires object detection label to be a list of torch.Tensor')
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
            raise ValidationError('Check requires detection predictions to be a list with an entry for each'
                                  ' sample')
        if len(batch_predictions) == 0:
            raise ValidationError('Check requires detection predictions to be a non-empty list')
        if not isinstance(batch_predictions[0], torch.Tensor):
            raise ValidationError('Check requires detection predictions to be a list of torch.Tensor')
        if len(batch_predictions[0].shape) != 2:
            raise ValidationError('Check requires detection predictions to be a list of 2D tensors')
        if batch_predictions[0].shape[1] != 6:
            raise ValidationError('Check requires detection predictions to be a list of 2D tensors, when '
                                  'each row has 6 columns: [x, y, width, height, class_probability, class_id]')
