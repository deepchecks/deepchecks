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
"""Utils module with methods for fast calculations."""
import typing as t

import numpy as np

from deepchecks.core.errors import DeepchecksValueError


def fast_sum_by_row(matrix: np.ndarray) -> np.array:
    """Faster alternative to np.sum(matrix, axis=1)."""
    return np.matmul(matrix, np.ones(matrix.shape[1]))


def sequence_to_numpy(sequence: t.Sequence):
    """Convert a sequence into a numpy array."""
    if isinstance(sequence, np.ndarray):
        return sequence.flatten()
    elif isinstance(sequence, t.List):
        return np.asarray(sequence).flatten()
    else:
        raise DeepchecksValueError('Trying to convert a non sequence into a flat list.')
