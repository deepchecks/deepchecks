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
import imgaug


__all__ = ['set_seeds', 'apply_to_tensor']


def set_seeds(seed: int):
    """Set seeds for reproducibility.

    Imgaug uses numpy's State
    Albumentation uses Python and imgaug seeds

    Parameters
    ----------
    seed : int
        Seed to be set
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        imgaug.seed(seed)


T = t.TypeVar('T')


def apply_to_tensor(
    x: T,
    fn: t.Callable[[torch.Tensor], torch.Tensor]
) -> T:
    """Apply provided function to tensor instances recursivly."""
    if isinstance(x, torch.Tensor):
        return t.cast(T, fn(x))
    elif isinstance(x, (str, bytes, bytearray)):
        return x
    elif isinstance(x, (list, tuple, set)):
        return type(x)(apply_to_tensor(it, fn) for it in x)
    elif isinstance(x, dict):
        return type(x)((k, apply_to_tensor(v, fn)) for k, v in x.items())
    return x
