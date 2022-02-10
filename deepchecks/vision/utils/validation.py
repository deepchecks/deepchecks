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
"""Module for validation of the vision module."""
import random
import typing as t

import numpy as np
import torch

from deepchecks.core import errors
from deepchecks import vision  # pylint: disable=unused-import, is used in type annotations


__all__ = ['validate_model', 'set_seeds']


def validate_model(dataset: 'vision.VisionData', model: t.Any):
    """Receive a dataset and a model and check if they are compatible.

    Parameters
    ----------
    dataset : VisionData
        Built on a dataloader on which the model can infer.
    model : Any
        Model to be validated

    Raises
    ------
    DeepchecksValueError
        If the dataset and the model are not compatible
    """
    try:
        model(next(iter(dataset.get_data_loader()))[0])
    except Exception as exc:
        raise errors.ModelValidationError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )


def set_seeds(seed: int):
    """Set seeds for reproducibility.

    Imgaug uses numpy's State
    Albumentation uses Python and imgaug seeds

    Parameters
    ----------
    seed : int
        Seed to be set
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
