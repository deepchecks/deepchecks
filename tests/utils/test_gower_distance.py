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
import gower
import numpy as np
import pandas as pd
from hamcrest import (assert_that, greater_than, has_item, has_length,
                      less_than_or_equal_to)

from deepchecks.utils import gower_distance


def test_mix_columns():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
                         'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]})
    # Act
    dist, _ = gower_distance.calculate_nearest_neighbours_distances(data[['col1']], data[['col2']], 5)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[3], has_item(greater_than(0.49)))
    assert_that(max(dist[0]), less_than_or_equal_to(0))


def test_mix_columns_full_matrix():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
                         'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]})
    is_categorical_arr = np.array([True, False], dtype=bool)
    # Act
    dist = gower_distance.gower_matrix(data=np.asarray(data),
                                       cat_features=is_categorical_arr)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[-1], has_length(data.shape[0]))
    assert_that(dist[3], has_item(greater_than(0.01)))
    assert_that(min(dist[0]), less_than_or_equal_to(0))


def test_mix_columns_full_matrix_with_nulls():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'b', 'a', 'a', None, 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
                         'col2': [1, 1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1, 1000]})
    is_categorical_arr = np.array([True, False], dtype=bool)
    # Act
    dist = gower_distance.gower_matrix(data=np.asarray(data),
                                       cat_features=is_categorical_arr)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[-1], has_length(data.shape[0]))
    assert_that(dist[3], has_item(greater_than(0.01)))
    assert_that(min(dist[0]), less_than_or_equal_to(0))


def test_mix_columns_nn_matrix_with_nulls_vectorized():
    # Arrange
    data = pd.DataFrame({'col1': ['a', 'a', 'a', 'b', 'a', 'a', None, 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
                         'col2': [1, 1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1, 1000]})
    # Act
    dist, _ = gower_distance.calculate_nearest_neighbours_distances(data[['col1']], data[['col2']], 3)
    # Assert
    assert_that(dist[-1], has_item(1))
    assert_that(dist[3], has_item(greater_than(0.4)))
    assert_that(min(dist[0]), less_than_or_equal_to(0))


def test_compare_other_package_iris(iris_dataset):
    data = iris_dataset.data
    data.drop_duplicates(inplace=True)
    dist, _ = gower_distance.calculate_nearest_neighbours_distances(data[iris_dataset.cat_features],
                                                                    data[iris_dataset.numerical_features], 3)
    dist = dist.round(5).astype(np.float32)
    for i in range(data.shape[0]):
        closest_to_i = gower.gower_topn(data.iloc[i:i + 1, :4], data.iloc[:, :4], n=3)
        assert (closest_to_i['values'].round(5) == dist[i, :]).all()
