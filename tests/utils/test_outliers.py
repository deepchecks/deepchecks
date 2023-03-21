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
import numpy as np
from hamcrest import assert_that, calling, equal_to, raises

from deepchecks.core.errors import DeepchecksValueError, NotEnoughSamplesError
from deepchecks.utils.outliers import iqr_outliers_range, sharp_drop_outliers_range


def test_iqr_range():
    data = np.array(list(range(10)))

    assert_that(iqr_outliers_range(data, (25, 75), 1), equal_to((-2.25, 11.25)))
    assert_that(iqr_outliers_range(data, (25, 75), 1.5), equal_to((-4.5, 13.5)))
    assert_that(iqr_outliers_range(data, (10, 60), 0.8), equal_to((-2.6999999999999997, 9.0)))

    assert_that(calling(iqr_outliers_range).with_args(data, (.25, .75), 1),
                raises(DeepchecksValueError, 'IQR range must contain two numbers between 0 to 100'))


def test_sharp_drop_outliers_range():
    data_counts = np.array([0.6, 0.37, 0.02, 0.005, 0.003, 0.002])  # sharp drop is between 0.37 and 0.02

    assert_that(sharp_drop_outliers_range(data_counts), equal_to(0.02))
    assert_that(sharp_drop_outliers_range(data_counts, 0.99), equal_to(None))
    assert_that(sharp_drop_outliers_range(data_counts, 0.4), equal_to(0.02))
    assert_that(sharp_drop_outliers_range(data_counts, 0.3, max_outlier_percentage=0.05), equal_to(0.02))
    assert_that(sharp_drop_outliers_range(data_counts, 0.3, max_outlier_percentage=0.5), equal_to(0.37))

    assert_that(calling(sharp_drop_outliers_range).with_args([0.1, 0.3]),
                raises(DeepchecksValueError, 'Data percents must sum to 1'))

