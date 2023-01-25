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
try:
    import torch  # noqa: F401
except ImportError as error:
    raise ImportError('PyTorch is not installed. Please install torch '
                      'in order to use deepchecks vision.') from error
from deepchecks.vision import metrics
from deepchecks.vision.base_checks import ModelOnlyCheck, SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.suite import Suite
from deepchecks.vision.vision_data import VisionData
from deepchecks.vision.vision_data.simple_classification_data import classification_dataset_from_directory
from deepchecks.vision.vision_data.utils import BatchOutputFormat

__all__ = [
    'VisionData',
    'BatchOutputFormat',
    'classification_dataset_from_directory',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'Suite',
    'metrics',
]
