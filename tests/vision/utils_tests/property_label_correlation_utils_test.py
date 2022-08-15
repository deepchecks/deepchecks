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
"""Test for the property label correlation utils module."""
import pandas as pd
from hamcrest import assert_that, equal_to

from deepchecks.vision.utils.property_label_correlation_utils import is_float_column


def test_is_float_column():
    col = pd.Series([1, 2, 3, 4, 5])
    assert_that(is_float_column(col), equal_to(False))

    col = pd.Series(['a', 'b', 'c'])
    assert_that(is_float_column(col), equal_to(False))

    col = pd.Series(['a', 'b', 5.5])
    assert_that(is_float_column(col), equal_to(False))

    col = pd.Series([1, 2, 3, 4, 5], dtype='float')
    assert_that(is_float_column(col), equal_to(False))

    col = pd.Series([1, 2, 3, 4, 5.5], dtype='float64')
    assert_that(is_float_column(col), equal_to(True))
