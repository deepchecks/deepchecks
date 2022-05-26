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
import numpy as np
from hamcrest import assert_that, calling, close_to, has_items, has_length, instance_of, raises
from scipy.special import softmax
from sklearn.metrics import log_loss

from deepchecks import CheckFailure
from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksProcessError, DeepchecksValueError
from deepchecks.tabular.checks.model_evaluation.model_error_analysis import ModelErrorAnalysis
from deepchecks.utils.single_sample_metrics import per_sample_cross_entropy
from tests.base.utils import equal_condition_result


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
               'There is no label defined to use on the dataset')
    )


def test_model_error_analysis_regression_not_meaningful(diabetes_split_dataset_and_model):
    # Arrange
    train, val, model = diabetes_split_dataset_and_model

    # Assert
    assert_that(ModelErrorAnalysis().run(train, val, model), instance_of(CheckFailure))


def test_model_error_analysis_classification(iris_labeled_dataset, iris_adaboost):
    # Act
    result_value = ModelErrorAnalysis().run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value

    # Assert
    assert_that(result_value['feature_segments']['petal length (cm)'], has_length(2))


def test_binary_string_model_info_object(iris_binary_string_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = iris_binary_string_split_dataset_and_model

    # Assert
    assert_that(ModelErrorAnalysis().run(train_ds, test_ds, clf), instance_of(CheckFailure))


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
            details='Accuracy difference for failed features: {\'petal length (cm)\': \'10.91%\', '
                    '\'petal width (cm)\': \'8.33%\'}',
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
            details='Average Accuracy difference: 9.62%',
            name='The performance difference of the detected segments must not be greater than 200%',
        )
    ))


def test_per_sample_log_loss():
    np.random.seed(0)
    n_classes = 2
    n_samples = 1000
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randn(n_samples, n_classes)
    y_pred = softmax(y_pred, axis=1)

    assert_that(per_sample_cross_entropy(y_true, y_pred).mean(), close_to(log_loss(y_true, y_pred), 1e-10))

    n_classes = 10
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randn(n_samples, n_classes)
    y_pred = softmax(y_pred, axis=1)

    assert_that(per_sample_cross_entropy(y_true, y_pred).mean(), close_to(log_loss(y_true, y_pred), 1e-10))
