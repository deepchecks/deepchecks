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
"""Test drift utils"""
import numpy as np
from hamcrest import assert_that, calling, close_to, equal_to, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.drift import cramers_v, earth_movers_distance


def test_emd():
    dist1 = np.ones(100)
    dist2 = np.zeros(100)
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, equal_to(1))


def test_real_input():
    # Move half of the dirt (0-50) to 2/3 of the distance (100-150) with the middle (50-100) staying unmoved.
    # Therefore, result should be 1/2 * 2/3 = 1/3
    dist1 = np.array(range(100))
    dist2 = np.array(range(50, 150))
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, close_to(0.33, 0.01))


def test_emd_scaling():
    dist1 = np.ones(100) * 10
    dist2 = np.zeros(100)
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, equal_to(1))


def test_emd_margin_filter():
    dist1 = np.concatenate([np.ones(99) * 10, np.ones(1) * 100])
    dist2 = np.concatenate([np.zeros(99), np.ones(1)])
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0.01)
    assert_that(res, equal_to(1))


def test_emd_raises_exception():
    dist1 = np.ones(100)
    dist2 = np.zeros(100)
    assert_that(
        calling(earth_movers_distance).with_args(dist1, dist2, -1),
        raises(DeepchecksValueError, r'margin_quantile_filter expected a value in range \[0, 0.5\), instead got -1')
    )


def test_cramers_v():
    dist1 = np.array(['a'] * 200 + ['b'] * 800)
    dist2 = np.array(['a'] * 400 + ['b'] * 600)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.21, 0.01))


def test_cramers_v_completely_diff_columns():
    dist1 = np.array(['a'] * 1000)
    dist2 = np.array(['b'] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(1, 0.01))


def test_cramers_v_single_value_columns():
    dist1 = np.array(['a'] * 1000)
    dist2 = np.array(['a'] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, equal_to(0))


def test_cramers_v_with_nones():
    dist1 = np.array(['a'] * 200 + ['b'] * 800 + [None] * 100)
    dist2 = np.array(['a'] * 400 + ['b'] * 600)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.3, 0.01))


def test_cramers_v_min_category_ratio():
    dist1 = np.array(['a'] * 200 + ['b'] * 800 + ['c'] * 10 + ['d'] * 10)
    dist2 = np.array(['a'] * 400 + ['b'] * 620)
    res = cramers_v(dist1=dist1, dist2=dist2, min_category_size_ratio=0)
    assert_that(res, close_to(0.228, 0.01))
    res_min_cat_ratio = cramers_v(dist1=dist1, dist2=dist2, min_category_size_ratio=0.1)
    assert_that(res_min_cat_ratio, close_to(0.208, 0.01))
