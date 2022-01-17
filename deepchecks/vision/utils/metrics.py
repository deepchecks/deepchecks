import enum

from torch import nn

from deepchecks.vision.utils import validation
from deepchecks.vision import VisionDataset
from deepchecks.utils.typing import BasicModel
from ignite.metrics import Accuracy, Precision, Recall, IoU, mIoU

__all__ = [
    'task_type_check',
    'ModelType'
]


class ModelType(enum.Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'


DEFAULT_CLASSIFICATION_SCORERS = {
    'Accuracy': Accuracy(),
    'Precision': Precision(),
    'Recall':  Recall()
}


# DEFAULT_OBJECT_DETECTION_SCORERS = {
#     'IoU': IoU(),
#     'mIoU': mIoU()
# }


def task_type_check(
    model: nn.Module,
    dataset: VisionDataset
) -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (BasicModel): Model object - used to check if it has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    validation.model_type_validation(model)
    dataset.validate_label()

    # Trying to infer the task type from the label shape
    label_shape = dataset.get_label_shape()

    # Means the tensor is an array of scalars
    if len(label_shape) == 1:
        return ModelType.CLASSIFICATION





