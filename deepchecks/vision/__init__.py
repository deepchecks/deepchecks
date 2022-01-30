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
"""Package for vision functionality."""
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
    import torch  # noqa: F401
    import torchvision  # noqa: F401

except ImportError as torch_missing:
    raise ImportError("PyTorch is not installed. Please install torch and torchvision "
                      "in order to use deepchecks.vision functionalities.") from torch_missing

__all__ = [
    "VisionDataset",
    "Context",
    "Check",
    "SingleDatasetBaseCheck",
    "TrainTestBaseCheck",
    "ModelOnlyBaseCheck",
    "Suite"
]
