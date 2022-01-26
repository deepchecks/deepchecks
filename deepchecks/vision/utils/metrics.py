import enum
import typing as t

import torch
from ignite.engine import Engine
from torch import nn

from deepchecks import errors
from deepchecks.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.utils import validation
from deepchecks.vision import VisionDataset
from deepchecks.utils.typing import BasicModel
from ignite.metrics import Precision, Recall, IoU, mIoU, Metric, ConfusionMatrix
from .accuracy_multiclass import Accuracy

__all__ = [
    'task_type_check'
]

from .detection_precision_recall import DetectionPrecisionRecall

from ..base.vision_dataset import TaskType


def get_default_classification_scorers(n_classes: int):
    return {
        'Precision': Precision(),
        'Recall': Recall()
    }


def get_default_object_detection_scorers(n_classes: int):
    return {
        'mAP': DetectionPrecisionRecall()
    }


def get_default_semantic_segmentation_scorers(n_classes: int):
    return {
        'IoU': IoU(ConfusionMatrix(num_classes=n_classes)),
        'mIoU': mIoU(ConfusionMatrix(num_classes=n_classes))
    }


def task_type_check(
        model: nn.Module,
        dataset: VisionDataset
) -> TaskType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (BasicModel): Model object - used to check if it has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    validation.model_type_validation(model)
    dataset.validate_label()

    return TaskType(dataset.label_type)


def get_scorers_list(
        model,
        dataset: VisionDataset,
        n_classes: int,
        alternative_scorers: t.List[Metric] = None,
        multiclass_avg: bool = True
) -> t.Dict[str, Metric]:
    task_type = task_type_check(model, dataset)

    if alternative_scorers:
        # Validate that each alternative scorer is a correct type
        for met in alternative_scorers:
            if not isinstance(met, Metric):
                raise DeepchecksValueError("alternative_scorers should contain metrics of type ignite.Metric")
        scorers = alternative_scorers
    elif task_type == TaskType.CLASSIFICATION:
        scorers = get_default_classification_scorers(n_classes)
    elif task_type == TaskType.OBJECT_DETECTION:
        scorers = get_default_object_detection_scorers(n_classes)
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        scorers = get_default_object_detection_scorers(n_classes)
    else:
        raise DeepchecksNotSupportedError(f'No scorers match task_type {task_type}')

    return scorers


def calculate_metrics(metrics: t.List[Metric], dataset: VisionDataset, model: nn.Module,
                      prediction_extract: t.Callable = None) -> t.Dict[str, float]:
    """Calculate a list of ignite metrics on a given model and dataset.

    Args:
        metrics: (List[Metric]) List of metrics to calculate
        dataset (VisionDataset): dataset object to calculate metrics on
        model (nn.Module): Model object
        prediction_extract: (Callable) Function to convert the model output to the appropriate format for the label type

    Returns:
        Dict of metrics with the name as key and the value as value

    """

    def process_function(_, batch):
        X = batch[0]
        Y = batch[1]

        predictions = model.forward(X)

        if prediction_extract:
            predictions = prediction_extract(predictions)

        return predictions, Y

    engine = Engine(process_function)
    for metric in metrics:
        metric.attach(engine, type(metric).__name__)

    state = engine.run(dataset.get_data_loader())

    results = {k: v.tolist() for k, v in state.metrics.items()}
    return results
