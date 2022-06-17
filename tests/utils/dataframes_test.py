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
"""Tests for dataframes"""
import pandas as pd
from hamcrest import assert_that, close_to, contains_exactly
from scipy.stats import pearsonr

from deepchecks.utils.dataframes import generalized_corrwith


def test_generalized_corrwith():
    df1 = pd.DataFrame({'blue': [1, 2, 3, 4, 5], 'yellow': [1, 2, 2, 4, -1], 'red': [9, 2, 9, 9, 9]})
    df2 = pd.DataFrame({'cat': [1, 2, 3, 4, 5], 'dog': [1, 0, 0, 4, 5]})
    result = generalized_corrwith(df1, df2, lambda x, y: pearsonr(x, y)[0])
    assert_that(result.loc['blue', 'cat'], close_to(1.0, 0.01))
    assert_that(result.index, contains_exactly('blue', 'yellow', 'red'))
    assert_that(result.columns, contains_exactly('cat', 'dog'))
