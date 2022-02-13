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
from collections import Counter
from typing import Union, Callable, Optional

__all__ = ['DetectionLabelFormatter', 'DetectionPredictionFormatter']

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base_formatters import BaseLabelFormatter, BasePredictionFormatter
from ...core.errors import DeepchecksValueError


class DetectionLabelFormatter(BaseLabelFormatter):
    """
    Class for encoding the detection annotations to the required format.

    Parameters
    ----------
    label_formatter : Union[str, Callable]
        Function or string that specifies the formatting function.
        If a string, it must be one of the following:
            - 'xyxy' - x1, y1, x2, y2 represent the upper left and lower right corners of the bounding box.
            - 'xywh' - x1, y1, w, h represent the upper left corner of the bounding box and its width and height.
            - 'cxcywh' - cx, cy, w, h represent the center of the bounding box and its width and height.
            - 'xyxyn' - x1, y1, x2, y2, n represent the upper left and lower right corners of the bounding box,
                        normalized to the image dimensions.
            - 'xywhn' - x1, y1, w, h, represent the upper left corner of the bounding box and its width and height,
                        normalized by the image dimensions.
            - 'cxcywhn' - (YOLO format) x, y, w, h, represent the center of the bounding box and its width and height,
                          normalized by the image dimensions.
        In addition, the label shape should be a list of length N containing tensors of shape (M, 5), where N is the
        number of samples, M is the number of bounding boxes, and each bounding box is represented by 5 values:
        (class_id, 4 coordinates in the format specified by the `label_formatter` parameter).

        If a function, it must follow the signature:
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        List of length N containing tensors of shape (M, 5), where N is the number of samples,
        M is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
        x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
        the width and height of the bounding box (in pixels) and class_id is the class id.

    Examples
    --------
    >>> import torch
    >>> from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
    >>> def yolo_to_coco(input_batch_from_loader):
    ...     return [torch.stack([torch.cat((bbox[1:3], bbox[4:] - bbox[1:3], bbox[0]), dim=0) for bbox in image])
    ...             for image in input_batch_from_loader]
    >>> label_formatter = DetectionLabelFormatter(yolo_to_coco)

    Or, if your labels are in a common format:

    >>> from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
    >>> label_formatter = DetectionLabelFormatter('cxcywhn')

    See Also
    --------
    DetectionPredictionFormatter

    """

    label_formatter: Union[str, Callable]

    def __init__(self, label_formatter: Union[str, Callable]):
        super().__init__(label_formatter)
        self.label_formatter = label_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        if isinstance(self.label_formatter, Callable):
            return self.label_formatter(*args, **kwargs)
        elif isinstance(self.label_formatter, str):
            pass

    def get_samples_per_class(self, data_loader: DataLoader):
        """
        Get the number of samples per class.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to get the samples per class from.

        Returns
        -------
        Counter
            Counter of the number of samples per class.
        """
        counter = Counter()
        for batch in data_loader:
            list_of_arrays = self(batch[1])
            class_list = sum([arr.reshape((-1, 5))[:, 0].tolist() for arr in list_of_arrays], [])
            counter.update(class_list)

        return counter

    def validate_label(self, data_loader: DataLoader) -> Optional[str]:
        """
        Validate the label.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to get the samples per class from.

        Returns
        -------
        Optional[str]
            None if the label is valid, otherwise a string containing the error message.

        """
        batch = next(iter(data_loader))
        if len(batch) != 2:
            return 'Check requires dataset to have a label'

        label_batch = self(batch[1])
        if not isinstance(label_batch, list):
            return 'Check requires object detection label to be a list with an entry for each sample'
        if len(label_batch) == 0:
            return 'Check requires object detection label to be a non-empty list'
        if not isinstance(label_batch[0], (torch.Tensor, np.ndarray)):
            return 'Check requires object detection label to be a list of torch.Tensor or numpy array'
        if len(label_batch[0].shape) != 2:
            return 'Check requires object detection label to be a list of 2D tensors'
        if label_batch[0].shape[1] != 5:
            return 'Check requires object detection label to be a list of 2D tensors, when ' \
                   'each row has 5 columns: [class_id, x, y, width, height]'


class DetectionPredictionFormatter(BasePredictionFormatter):
    """
    Class for encoding the detection prediction to the required format.

    Parameters
    ----------
    prediction_formatter : Callable
        Function that takes in a batch of predictions and returns the encoded labels in the following format:
        List of length N containing tensors of shape (B, 6), where N is the number of images,
        B is the number of bounding boxes detected in the sample and each bounding box is represented by 6 values:
        [x, y, w, h, confidence, class_id]. x and y are the coordinates (in pixels) of the upper left corner of the
        bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the confidence of
        the model and class_id is the class id.

    Examples
    --------
    >>> import torch
    >>> import typing as t
    >>> from deepchecks.vision.utils.detection_formatters import DetectionPredictionFormatter
    >>> def yolo_wrapper(
    ...     predictions: 'ultralytics.models.common.Detections'  # noqa: F821
    ... ) -> t.List[torch.Tensor]:
    ...     return_list = []
    ...
    ...     # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
    ...     for single_image_tensor in predictions.pred:
    ...         pred_modified = torch.clone(single_image_tensor)
    ...         pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
    ...         pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top
    ...         return_list.append(pred_modified)
    ...
    ...     return return_list

    >>> label_formatter = DetectionLabelFormatter(yolo_wrapper)


    See Also
    --------
    DetectionLabelFormatter

    """

    def __init__(self, prediction_formatter: Callable):
        super().__init__(prediction_formatter)
        self.prediction_formatter = prediction_formatter

    def __call__(self, *args, **kwargs):
        """Call the encoder."""
        return self.prediction_formatter(*args, **kwargs)

    def validate_prediction(self, batch_predictions, n_classes: int, eps: float = 1e-3):
        """
        Validate the prediction.

        Parameters
        ----------
        batch_predictions : t.Any
            Model prediction for a batch (output of model(batch[0]))
        n_classes : int
            Number of classes.
        eps : float , default: 1e-3
            Epsilon value to be used in the validation, by default 1e-3
        """
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
