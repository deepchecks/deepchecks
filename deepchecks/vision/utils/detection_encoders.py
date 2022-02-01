from typing import Union, Callable

__all__ = ["DetectionLabelEncoder", "DetectionPredictionEncoder"]


class DetectionLabelEncoder:
    """
    Class for encoding the detection annotations to the required format.

    Parameters
    ----------
    label_encoder : Union[str, Callable]
        Function or string that specifies the encoding function.
        If a string, it must be one of the following:
            - 'xyxy' - x1, y1, x2, y2 represent the upper left and lower right corners of the bounding box.
            - 'xywh' - x1, y1, w, h represent the upper left corner of the bounding box and its width and height.
            - 'cxcywh' - x, y, w, h represent the center of the bounding box and its width and height.
            - 'xyxyn' - x1, y1, x2, y2, n represent the upper left and lower right corners of the bounding box,
                        normalized to the image dimensions.
            - 'xywhn' - x1, y1, w, h, represent the upper left corner of the bounding box and its width and height,
                        normalized by the image dimensions.
            - 'cxcywhn' - (YOLO format) x, y, w, h, represent the center of the bounding box and its width and height,
                          normalized by the image dimensions.
        In addition, the label shape should be a list of length N containing tensors of shape (M, 5), where N is the
        number of samples, M is the number of bounding boxes, and each bounding box is represented by 5 values:
        (class_id, 4 coordinates in the format specified by the `label_encoder` parameter).

        If a function, it must follow the signature:
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        List of length N containing tensors of shape (M, 5), where N is the number of samples,
        M is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
        x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
        the width and height of the bounding box (in pixels) and class_id is the class id.

    """

    label_encoder: Union[str, Callable]

    def __init__(self, label_encoder: Union[str, Callable]):
        self.label_encoder = label_encoder

    def __call__(self, *args, **kwargs):
        if isinstance(self.label_encoder, Callable):
            return self.label_encoder(*args, **kwargs)
        elif isinstance(self.label_encoder, str):
            pass


class DetectionPredictionEncoder:
    """
    Class for encoding the detection prediction to the required format.

    Parameters
    ----------
    prediction_encoder : Callable
        Function that takes in a batch of labels and returns the encoded labels in the following format:
        List of length N containing tensors of shape (B, 6), where N is the number of images,
        B is the number of bounding boxes detected in the sample and each bounding box is represented by 6 values:
        [x, y, w, h, confidence, class_id]. x and y are the coordinates (in pixels) of the upper left corner of the
        bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the confidence of
        the model and class_id is the class id.
    """

    def __init__(self, prediction_encoder: Callable):
        self.prediction_encoder = prediction_encoder

    def __call__(self, *args, **kwargs):
        return self.prediction_encoder(*args, **kwargs)
