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
"""Contains unit tests for the outlier_sample_detection check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, greater_than, has_item, has_items, has_length, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks import OutlierSampleDetection
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.datasets.regression import avocado
from tests.base.utils import equal_condition_result


def test_condition_input_validation():
    # Assert
    assert_that(
        calling(OutlierSampleDetection().add_condition_outlier_ratio_less_or_equal).with_args(max_outliers_ratio=-1),
        raises(DeepchecksValueError, 'max_outliers_ratio must be between 0 and 1'))


def test_check_input_validation():
    # Assert
    assert_that(
        calling(OutlierSampleDetection).with_args(extent_parameter=-1),
        raises(DeepchecksValueError, 'extend_parameter must be a positive integer'))


def test_condition_with_argument():
    # Arrange
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=[])
    # Act
    check = OutlierSampleDetection(nearest_neighbors_percent=0.2).add_condition_outlier_ratio_less_or_equal(0.1)
    result = check.conditions_decision(check.run(dataset))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='8.3% of dataset samples above outlier threshold',
                               name='Ratio of samples exceeding the outlier score threshold 0.7 is less or equal to '
                                    '10%')
    ))


def test_condition_without_argument():
    # Arrange
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=[])
    # Act
    check = OutlierSampleDetection(nearest_neighbors_percent=0.2).add_condition_no_outliers()
    result = check.conditions_decision(check.run(dataset))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='0% of dataset samples above outlier threshold',
                               name='No samples in dataset over outlier score of 0.7')
    ))


def test_integer_single_column_no_nulls():
    # Arrange
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=[])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.2).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.76)))


def test_integer_single_column_with_nulls():
    # Arrange
    data = {'col1': [1, 1, 1, None, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=[])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.3).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.44)))
    assert_that(np.unique(result.value), has_length(2))


def test_integer_columns_with_nulls():
    # Arrange
    data = {'col1': [1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1, 1, 1, 1000],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=[])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.3).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.44)))


def test_single_column_cat_no_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b']}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.3).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.57)))


def test_mix_types_no_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b'],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.3).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.57)))


def test_mix_types_with_nulls():
    # Arrange
    data = {'col1': ['a', 'a', 'a', 'a', 'a', pd.NA, 'a', 'a', 'a', 'a', np.nan, 'a', 'a', 'b'],
            'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]}
    dataset = Dataset(pd.DataFrame(data=data), cat_features=['col1'])
    # Act
    result = OutlierSampleDetection(nearest_neighbors_percent=0.3).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.42)))


def test_iris_regular(iris_dataset):
    # Act
    result = OutlierSampleDetection(n_to_show=2).run(iris_dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.91)))


def test_iris_modified(iris):
    # Arrange
    iris_modified = iris.copy()
    iris_modified.loc[len(iris.index)] = [1, 10, 1000, 1000, 1]
    # Act
    result = OutlierSampleDetection(n_to_show=2).run(Dataset(iris_modified))
    # Assert
    assert_that(result.value, has_item(greater_than(0.99)))


def test_avocado():
    # Arrange
    dataset = avocado.load_data()[0]
    # Act
    result = OutlierSampleDetection(extent_parameter=3, n_samples=5000, timeout=0).run(dataset)
    # Assert
    assert_that(result.value, has_item(greater_than(0.67)))
