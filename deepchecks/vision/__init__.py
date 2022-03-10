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
"""Package for vision functionality."""
import logging
from .vision_data import VisionData
from .classification_data import ClassificationData
from .detection_data import DetectionData
from .base import (
    Context,
    Suite,
    SingleDatasetCheck,
    TrainTestCheck,
    ModelOnlyCheck,
    Batch,
)

logger = logging.getLogger("deepchecks")

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError as error:
    raise ImportError("PyTorch is not installed. Please install torch and torchvision "
                      "in order to use deepchecks.vision functionalities.") from error

__all__ = [
    "VisionData",
    "ClassificationData",
    "DetectionData",
    "Context",
    "SingleDatasetCheck",
    "TrainTestCheck",
    "ModelOnlyCheck",
    "Suite",
    "Batch"
]
