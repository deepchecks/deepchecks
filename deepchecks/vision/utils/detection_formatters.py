# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for defining detection encoders."""
from typing import Union, Callable
from itertools import chain
import numpy as np
import torch

from .base_formatters import BaseLabelFormatter, BasePredictionFormatter
from deepchecks.core.errors import DeepchecksValueError

__all__ = ['DetectionLabelFormatter', 'DetectionPredictionFormatter']


class DetectionLabelFormatter(BaseLabelFormatter):
    """
    Class for encoding the detection annotations to the required format.

    Parameters
    ----------
    label_formatter : Callable
        Function that takes in a batch from DataLoader and returns only the encoded labels in the following format:
        List of length N containing tensors of shape (M, 5), where N is the number of samples,
        M is the number of bounding boxes in the sample and each bounding box is represented by 5 values: **(class_id,
        x, y, w, h)**. x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
        the width and height of the bounding box (in pixels) and class_id is the class id.

    Examples
    --------
    >>> import torch
    ... from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
    ...
    ...
    ... def yolo_to_coco(batch):
    ...     return [torch.stack(
    ...             [torch.cat((bbox[1:3], bbox[4:] - bbox[1:3], bbox[0]), dim=0)
    ...                 for bbox in image])
    ...             for image in batch[1]]
    ...
    ...
    ... label_formatter = DetectionLabelFormatter(yolo_to_coco)


    See Also
    --------
    DetectionPredictionFormatter

    """

    label_formatter: Union[str, Callable]

    def __init__(self, label_formatter: Union[str, Callable] = lambda x: x[1]):
        super().__init__(label_formatter)
        self.label_formatter = label_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        if isinstance(self.label_formatter, Callable):
            return self.label_formatter(*args, **kwargs)
        elif isinstance(self.label_formatter, str):
            pass

    def get_classes(self, batch_labels):
        """Get a labels batch and return classes inside it."""
        def get_classes_from_single_label(tensor):
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
        labels = self(batch)
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


class DetectionPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the detection prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch from DataLoader and model, and returns the encoded labels in the
        following format:
        List of length N containing tensors of shape (B, 6), where N is the number of images,
        B is the number of bounding boxes detected in the sample and each bounding box is represented by 6 values:
        **[x, y, w, h, confidence, class_id]**. x and y are the coordinates (in pixels) of the upper left corner of the
        bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the confidence of
        the model and class_id is the class id.

    Examples
    --------
    >>> import torch
    ... import typing as t
    ... from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
    ...
    ...
    ... def yolo_wrapper(
    ...     batch, model, device
    ... ) -> t.List[torch.Tensor]:
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
    ...
    ...
    ... label_formatter = DetectionPredictionFormatter(yolo_wrapper)


    See Also
    --------
    DetectionLabelFormatter

    """

    def __init__(self, prediction_formatter: Callable = None):
        super().__init__(prediction_formatter)
        if prediction_formatter is None:
            self.prediction_formatter = lambda batch, model, device: model.to(device)(batch[0].to(device))
        else:
            self.prediction_formatter = prediction_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.prediction_formatter(*args, **kwargs)

    def validate_prediction(self, batch, model, device, n_classes: int = None, eps: float = 1e-3):
        """
        Validate the prediction.

        Parameters
        ----------
        batch : t.Any
            Batch from DataLoader
        model : t.Any
        device : torch.Device
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
        batch_predictions = self(batch, model, device)
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
