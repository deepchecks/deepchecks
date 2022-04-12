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
import numpy as np
import pandas as pd
from hamcrest import assert_that, has_item, greater_than, less_than_or_equal_to

from deepchecks.utils import gower_distance


def test_integer_column_with_nulls():
    # Arrange
    data = pd.DataFrame({'col1': [1, 1, 1, None, 1, 1, 70, 1, 1, 1, 1, 1, 1000]})
    is_categorical_arr = np.array([False], dtype=bool)
    # Act
    dist, _ = gower_distance.gower_matrix_n_closets(data=np.asarray(data),
                                                    cat_features=is_categorical_arr,
                                                    num_neighbours=5)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert pd.isna(dist[3, 0])
    assert_that(max(dist[6]), less_than_or_equal_to(0.1))
    assert_that(max(dist[0]), less_than_or_equal_to(0))


def test_categorical_column_with_nulls():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'a', None, 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b']})
    is_categorical_arr = np.array([True], dtype=bool)
    # Act
    dist, _ = gower_distance.gower_matrix_n_closets(data=np.asarray(data),
                                                    cat_features=is_categorical_arr,
                                                    num_neighbours=5)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[4], has_item(1))
    assert_that(max(dist[0]), less_than_or_equal_to(0))


def test_mix_columns_with_nulls():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
                         'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]})
    is_categorical_arr = np.array([True, False], dtype=bool)
    # Act
    dist, _ = gower_distance.gower_matrix_n_closets(data=np.asarray(data),
                                                    cat_features=is_categorical_arr,
                                                    num_neighbours=5)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[3], has_item(greater_than(0.01)))
    assert_that(max(dist[0]), less_than_or_equal_to(0))
