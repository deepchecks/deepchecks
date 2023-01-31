# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Test preprocessing utils"""
import numpy as np
from hamcrest import assert_that, calling, equal_to, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.preprocessing import (OTHER_CATEGORY_NAME, preprocess_2_cat_cols_to_same_bins,
                                                         value_frequency)


def test_cat_cols_to_bins_no_max_num_categories():
    all_letters = list('abcde')
    dist1 = np.array(all_letters * 2)
    dist2 = np.array(all_letters * 3)
    res = preprocess_2_cat_cols_to_same_bins(dist1=dist1, dist2=dist2)

    assert_that(list(res[0]), equal_to([2] * 5))
    assert_that(list(res[1]), equal_to([3] * 5))
    assert_that(sorted(list(res[2])), equal_to(all_letters))


def test_cat_cols_to_bins_with_max_num_categories_and_sort_by_dist1():
    # Makes sure that doesn't ignore cats missing in test
    dist1 = np.array(list('aaabbbccd'))
    dist2 = np.array(list('aacdee'))
    res = preprocess_2_cat_cols_to_same_bins(dist1=dist1, dist2=dist2, max_num_categories=3, sort_by='dist1')

    assert_that(list(res[0]), equal_to([3, 3, 2, 1]))
    assert_that(list(res[1]), equal_to([2, 0, 1, 3]))
    assert_that(list(res[2]), equal_to(['a', 'b', 'c', OTHER_CATEGORY_NAME]))


def test_cat_cols_to_bins_with_max_num_categories_and_sort_by_diff():
    # Makes sure that doesn't ignore cats missing in test and train
    dist1 = np.array(list('aaabbbccd'))
    dist2 = np.array(list('aacdee'))
    res = preprocess_2_cat_cols_to_same_bins(dist1=dist1, dist2=dist2, max_num_categories=3, sort_by='difference',
                                             min_category_size_ratio=0)

    assert_that(list(res[0]), equal_to([3, 0, 3, 3]))
    assert_that(list(res[1]), equal_to([0, 2, 2, 2]))
    assert_that(list(res[2]), equal_to(['b', 'e', 'a', OTHER_CATEGORY_NAME]))


def test_cat_cols_to_bins_with_max_num_categories_and_sort_by_raises():
    dist1 = np.array(list('aaabbbccd'))
    dist2 = np.array(list('aacdee'))
    assert_that(
        calling(preprocess_2_cat_cols_to_same_bins).with_args(dist1=dist1, dist2=dist2, max_num_categories=3,
                                                              sort_by='bla'),
        raises(DeepchecksValueError, r'sort_by got unexpected value\: bla')
    )


def test_value_frequency():
    assert_that(value_frequency(np.array([1, 1, 1, 2, 2, 1, 3, 3, 3, 0])), equal_to([0.4, 0.2, 0.3, 0.1]))
    assert_that(value_frequency([1, np.NAN, 1,  1, 3, np.NAN, 3, 0]), equal_to([0.375, 0.25, 0.25, 0.125]))
