# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for segment performance check."""
from hamcrest import assert_that, has_entries, close_to, has_property, equal_to, calling, raises, is_, has_length, \
    has_items

from deepchecks import ConditionCategory
from deepchecks.errors import DeepchecksValueError, DeepchecksProcessError
from deepchecks.checks.performance.model_error_analysis import ModelErrorAnalysis
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(bad_dataset, bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(iris_dataset, iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


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
    assert_that(result_value['petal length (cm)'], has_length(2))


def test_condition_fail(iris_labeled_dataset, iris_adaboost):
    # Act
    check_result = ModelErrorAnalysis().add_condition_segments_ratio_performance_change_not_greater_than(
    ).run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=False,
            name='The percent change between the performance of detected segments must'
                 ' not exceed 5.00%',
            details='Segmentation of error by the features: petal length (cm), petal width (cm) resulted in percent '
                    'change in Accuracy (Default) larger than 5.00%.',
            category=ConditionCategory.WARN
        )
    ))


def test_condition_pass(iris_labeled_dataset, iris_adaboost):
    # Act
    check_result = ModelErrorAnalysis().add_condition_segments_ratio_performance_change_not_greater_than(2
                                                                                                         ).run(
        iris_labeled_dataset, iris_labeled_dataset, iris_adaboost)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The percent change between the performance of detected segments must'
                 ' not exceed 200%',
            category=ConditionCategory.WARN
        )
    ))
