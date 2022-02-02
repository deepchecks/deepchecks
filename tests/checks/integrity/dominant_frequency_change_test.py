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
"""
Contains unit tests for the dominant_frequency_change check
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from hamcrest import assert_that, calling, raises, equal_to, \
                     has_length, has_items, close_to, empty

from deepchecks.tabular import Dataset
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.integrity import DominantFrequencyChange

from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    cls = DominantFrequencyChange()
    assert_that(
        calling(cls.run).with_args(x, x),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str'
        )
    )


def test_no_leakage(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # Arrange
    check = DominantFrequencyChange(ratio_change_thres=3)
    # Act X
    result = check.run(train_dataset=train_ds, test_dataset=val_ds).value
    # Assert
    assert_that(result, empty())


def test_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                            features=iris_clean.feature_names,
                            label='target')

    test_df = pd.concat([x_test, y_test], axis=1)
    test_df.loc[test_df.index % 2 == 0, 'petal length (cm)'] = 5.1

    validation_dataset = Dataset(test_df,
                                 features=iris_clean.feature_names,
                                 label='target')
    # Arrange
    check = DominantFrequencyChange()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=validation_dataset).value
    row = result['petal length (cm)']
    # Assert
    assert_that(row['Value'], equal_to(5.1))
    assert_that(row['Train data %'], close_to(0.057, 0.001))
    assert_that(row['Test data %'], close_to(0.555, 0.001))
    assert_that(row['Train data #'], equal_to(6))
    assert_that(row['Test data #'], equal_to(25))
    assert_that(row['P value'], close_to(0, 0.00001))


def test_show_any(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # those params means any value should be included
    check = DominantFrequencyChange(dominance_ratio=0, ratio_change_thres=-1)
    # Act
    result = check.run(train_dataset=train_ds, test_dataset=val_ds).value
    # Assert
    assert_that(len(result), equal_to(len(train_ds.features)))


def test_show_none_dominance_ratio(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # because of dominance_ratio no value should be included
    check = DominantFrequencyChange(dominance_ratio=len(train_ds.features) + 1,
                                    ratio_change_thres=-1)
    # Act
    result = check.run(train_dataset=train_ds, test_dataset=val_ds).value
    # Assert
    assert_that(result, empty())


def test_show_none_ratio_change_thres(iris_split_dataset_and_model):
    train_ds, val_ds, _ = iris_split_dataset_and_model
    # because of ratio_change_thres no value should be included
    check = DominantFrequencyChange(dominance_ratio=0, ratio_change_thres=100)
    # Act
    result = check.run(train_dataset=train_ds, test_dataset=val_ds).value
    # Assert
    assert_that(result, empty())

def test_fi_n_top(diabetes_split_dataset_and_model):
    train, val, clf = diabetes_split_dataset_and_model
    # Arrange
    check = DominantFrequencyChange(dominance_ratio=0,
                                    ratio_change_thres=-1, n_top_columns=3)
    # Act
    result_ds = check.run(train, val, clf).display[1]
    # Assert
    assert_that(result_ds, has_length(3))


def test_condition_ratio_not_less_than_not_passed(iris_clean):
    # Arrange
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                            features=iris_clean.feature_names,
                            label='target')

    test_df = pd.concat([x_test, y_test], axis=1)

    # make duplicates in the test data
    test_df.loc[test_df.index % 2 == 0, 'petal length (cm)'] = 5.1
    test_df.loc[test_df.index / 3 > 8, 'sepal width (cm)'] = 2.7

    test_dataset = Dataset(test_df,
                           features=iris_clean.feature_names,
                           label='target')

    check = DominantFrequencyChange().add_condition_p_value_not_less_than(p_value_threshold = 0.0001)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='P value is not less than 0.0001',
                               details='Found columns with p-value below threshold: '
                                       '{\'sepal width (cm)\': \'7.63E-20\', \'petal length (cm)\': \'2.26E-11\'}'
    )))


def test_condition_ratio_not_less_than_passed(iris_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, _ = iris_split_dataset_and_model

    check = DominantFrequencyChange().add_condition_p_value_not_less_than()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='P value is not less than 0.0001')
    ))


def test_condition_ratio_of_change_not_greater_than_not_passed(iris_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, _ = iris_split_dataset_and_model

    check = DominantFrequencyChange().add_condition_ratio_of_change_not_greater_than(0.05)

    # Act
    result, *_ = check.conditions_decision(check.run(train_dataset=val_ds, test_dataset=train_ds))

    # Assert
    assert_that(result, equal_condition_result(
            is_pass=False,
            name='Change in ratio of dominant value in data is not greater than 5%',
            details='Found columns with % difference in dominant value above threshold: '
                    '{\'sepal width (cm)\': \'8%\'}'
    ))


def test_condition_ratio_of_change_not_greater_than_passed(iris_split_dataset_and_model):
    # Arrange
    train_ds, _, _ = iris_split_dataset_and_model

    check = DominantFrequencyChange().add_condition_ratio_of_change_not_greater_than()

    # Act
    result = check.conditions_decision(check.run(train_ds, train_ds))

    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Change in ratio of dominant value in data is not greater than 25%')
    ))