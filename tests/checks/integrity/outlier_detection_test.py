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
"""Contains unit tests for the outlier_detection check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, raises
from sklearn.datasets import load_iris

from deepchecks.tabular.checks import OutlierDetection
from deepchecks.tabular.dataset import Dataset


def test_condition_input_validation():
    # Assert
    assert_that(calling(OutlierDetection().add_condition_not_more_outliers_than).with_args(max_outliers_ratio=-1),
                raises(ValueError,
                       'max_outliers_ratio must be between 0 and 1'))


def test_integer_single_column_no_nulls():
    # Arrange
    data = {'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(dataframe)
    # Assert
    assert_that(max(result.value[0]) > 0.7)


def test_integer_columns_with_nulls():
    # Arrange
    data = {'col1': [1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1, 1, 1, 1000],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(dataframe)
    # Assert
    assert_that(max(result.value[0]) > 0.7)


def test_single_column_cat_no_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b']}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(dataset)
    # Assert
    assert_that(max(result.value[0]) > 0.7)


def test_mix_types_no_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(dataset)
    # Assert
    assert_that(max(result.value[0]) > 0.7)


def test_mix_types_with_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', pd.NA, 'a', 'a', 'a', 'a', np.nan, 'a', 'a', 'b'],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(dataset)
    # Assert
    assert_that(max(result.value[0]) > 0.7)


def test_iris_regular():
    # Arrange
    iris = load_iris()
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(pd.DataFrame(iris.data))
    # Assert
    print(max(result.value[0]))
    assert_that(max(result.value[0]) > 0.8)


def test_iris_modified():
    # Arrange
    iris = load_iris().data
    iris = np.vstack([iris, [0, 100, 10000, 100000]])
    # Act
    result = OutlierDetection(n_to_show=1, num_nearest_neighbors=5).run(pd.DataFrame(iris))
    # Assert
    print(max(result.value[0]))
    assert_that(max(result.value[0]) > 0.999)
