import logging
from .dataset import VisionDataset
from .base import (
    Context,
    Check,
    Suite,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck
)

logger = logging.getLogger("deepchecks")

try:
    import torch
except ImportError:
    logger.error("PyTorch is not installed. Please install it in order to use deepchecks.vision")

__all__ = [
    "VisionDataset",
    "Context",
    "Check",
    "SingleDatasetBaseCheck",
    "TrainTestBaseCheck",
    "ModelOnlyBaseCheck",
    "Suite"
]
