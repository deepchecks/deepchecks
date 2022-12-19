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
"""Module containing datasets and models for vision tasks."""
from . import classification, detection, segmentation

try:
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError as error:
    raise ImportError('PyTorch is not installed. Please install torch and torchvision '
                      'in order to use deepchecks.vision.datasets.') from error

__all__ = ['classification', 'detection', 'segmentation']
