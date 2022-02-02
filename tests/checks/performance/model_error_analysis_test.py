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
"""Tests for segment performance check."""
from hamcrest import assert_that, calling, raises, has_length, has_items

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, DeepchecksProcessError, DeepchecksNotSupportedError
from deepchecks.tabular.checks.performance.model_error_analysis import ModelErrorAnalysis
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(ModelErrorAnalysis().run).with_args(bad_dataset, bad_dataset, None),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(ModelErrorAnalysis().run).with_args(iris_dataset_no_label, iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?')
    )


def test_model_error_analysis_regression_not_meaningful(diabetes_split_dataset_and_model):
    # Arrange
    train, val, model = diabetes_split_dataset_and_model

    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(train, val, model),
                raises(DeepchecksProcessError,
                       'Unable to train meaningful error model'))


def test_model_error_analysis_classification(iris_labeled_dataset, iris_adaboost):
    # Act
    result_value = ModelErrorAnalysis().run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value

    # Assert
    assert_that(result_value['feature_segments']['petal length (cm)'], has_length(2))


def test_binary_string_model_info_object(iris_binary_string_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = iris_binary_string_split_dataset_and_model
    check = ModelErrorAnalysis()
    # Act X
    result_value = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result_value['feature_segments']['petal length (cm)'], has_length(2))


def test_condition_fail(iris_labeled_dataset, iris_adaboost):
    # Act
    check_result = ModelErrorAnalysis().add_condition_segments_performance_relative_difference_not_greater_than(
    ).run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=False,
            name='The performance difference of the detected segments must not be greater than 5%',
            details='Found change in Accuracy in features above threshold: {\'petal length (cm)\': \'10.91%\'}',
            category=ConditionCategory.WARN
        )
    ))


def test_condition_pass(iris_labeled_dataset, iris_adaboost):
    # Act
    condition_result = (
        ModelErrorAnalysis()
        .add_condition_segments_performance_relative_difference_not_greater_than(2)
        .run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost)
        .conditions_results
    )

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The performance difference of the detected segments must not be greater than 200%',
            category=ConditionCategory.WARN
        )
    ))
