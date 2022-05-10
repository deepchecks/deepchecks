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
"""Module containing usefull iter functions."""
import typing as t

import numpy as np

__all__ = ['flatten_matrix', 'join']


def flatten_matrix(matrix: np.ndarray) -> t.Iterator[t.Tuple[int, int, t.Any]]:
    """Flatten 2D matrix."""
    for row_index, row in enumerate(matrix):
        for column_index, cell in enumerate(row):
            yield row_index, column_index, cell


A = t.TypeVar('A')
B = t.TypeVar('B')


def join(l: t.List[A], item: B) -> t.Iterator[t.Union[A, B]]:
    """Concatenate a list of items into one iterator and put 'item' between elements of the list."""
    list_len = len(l) - 1
    for index, el in enumerate(l):
        yield el
        if index != list_len:
            yield item
