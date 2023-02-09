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
from hamcrest import assert_that, calling, has_items, raises, close_to

from deepchecks.core.check_result import CheckResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.tabular.base_checks import TrainTestCheck
from deepchecks.tabular.checks import SingleDatasetPerformance
from deepchecks.tabular.checks.model_evaluation.regression_error_distribution import RegressionErrorDistribution
from deepchecks.tabular.checks.model_evaluation.roc_report import RocReport
from deepchecks.tabular.checks.model_evaluation.simple_model_comparison import SimpleModelComparison
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.tabular.suites.default_suites import full_suite
from deepchecks.tabular.utils.task_type import TaskType
from tests.base.utils import equal_condition_result
from tests.common import get_expected_results_length, validate_suite_result
from tests.tabular.checks.model_evaluation.simple_model_comparison_test import assert_regression


def _dummify_model(train, test, model):
    y_pred_train = y_pred_test = y_proba_train = y_proba_test = None
    if hasattr(model, 'predict'):
        if train is not None:
            y_pred_train = model.predict(train.features_columns)
        if test is not None:
            y_pred_test = model.predict(test.features_columns)
    if hasattr(model, 'predict_proba'):
        if train is not None:
            y_proba_train = model.predict_proba(train.features_columns)
        if test is not None:
            y_proba_test = model.predict_proba(test.features_columns)

    return y_pred_train, y_pred_test, y_proba_train, y_proba_test


# copied from roc_report_test
def test_roc_condition_ratio_more_than_passed(iris_split_dataset_and_model):
    ds, _, clf = iris_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(ds, None, clf)

    check = RocReport().add_condition_auc_greater_than()
    result = check.conditions_decision(check.run(ds,
                                                 y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                 y_proba_train=y_proba_train, y_proba_test=y_proba_test))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='All classes passed, minimum AUC found is 1 for class 2.0',
                               name='AUC score for all the classes is greater than 0.7')
    ))


def test_can_run_classification_no_proba(iris_split_dataset_and_model):
    ds, _, clf = iris_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(ds, None, clf)
    y_proba_train = None
    y_proba_test = None

    check = SingleDatasetPerformance().add_condition_greater_than(0.9)
    result = check.conditions_decision(check.run(ds,
                                                 y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                 y_proba_train=y_proba_train, y_proba_test=y_proba_test))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for all of the metrics.',
                               name='Selected metrics scores are greater than 0.9')
    ))


def test_can_run_classification_no_proba_force_regression(iris_split_dataset_and_model):
    ds, _, clf = iris_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(ds, None, clf)
    y_proba_train = None
    y_proba_test = None

    ds = ds.copy(ds.data)
    ds._label_type = TaskType.REGRESSION
    check = SingleDatasetPerformance().add_condition_greater_than(0.9)
    result = check.conditions_decision(check.run(ds,
                                                 y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                 y_proba_train=y_proba_train, y_proba_test=y_proba_test))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Failed for metrics: [\'Neg RMSE\', \'Neg MAE\']',
                               name='Selected metrics scores are greater than 0.9')
    ))


# copied from regression_error_distribution_test
def test_regression_error_absolute_kurtosis_not_greater_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    test = Dataset(test.data.copy(), label='target', label_type='regression')
    test._data[test.label_name] =300
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(test, None, clf)

    check = RegressionErrorDistribution().add_condition_kurtosis_greater_than()

    # Act
    result = check.conditions_decision(check.run(test,
                                                 y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                                 y_proba_train=y_proba_train, y_proba_test=y_proba_test))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value higher than -0.1',
                               details='Found kurtosis value of -0.92572',
                               category=ConditionCategory.WARN)
    ))


#  copied from simple_model_comparison_test
def test_simple_model_comparison_regression_random_state(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(train_ds, test_ds, clf)
    check = SimpleModelComparison(strategy='uniform', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds,
                       y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                       y_proba_train=y_proba_train, y_proba_test=y_proba_test).value
    # Assert
    assert_regression(result)


#  copied from simple_model_comparison_test
def test_simple_model_comparison_regression_random_state_same_index(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    test_ds = test_ds.copy(test_ds.data.reset_index())
    train_ds = train_ds.copy(train_ds.data.reset_index())
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(train_ds, test_ds, clf)
    check = SimpleModelComparison(strategy='uniform', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds,
                       y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                       y_proba_train=y_proba_train, y_proba_test=y_proba_test).value
    # Assert
    assert_regression(result)


def test_new_data(diabetes_split_dataset_and_model):
    class NewDataCheck(TrainTestCheck):
        def run_logic(self, context: Context) -> CheckResult:
            model = context.model
            row = context.train.features_columns.head(1)
            row['s1'] = [0]
            return_value = model.predict(row)
            return CheckResult(return_value)
    # Arrange
    train, test, clf = diabetes_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(train, test, clf)

    # Act
    assert_that(
        calling(NewDataCheck().run)
            .with_args(train_dataset=train, test_dataset=test,
                       y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                       y_proba_train=y_proba_train, y_proba_test=y_proba_test),
        raises(
            DeepchecksValueError,
            r'Data that has not been seen before passed for inference with static predictions. Pass a real model to '
            r'resolve this')
    )

def test_bad_pred_shape(diabetes_split_dataset_and_model):
    # Arrange
    train, test, clf = diabetes_split_dataset_and_model
    y_pred_train, _, _, _ = _dummify_model(train, test, clf)

    # Act
    assert_that(
        calling(RegressionErrorDistribution().run)
            .with_args(dataset=test, y_pred=y_pred_train),
        raises(
            ValidationError,
            r'Prediction array expected to be of shape \(146,\) but was: \(296,\)')
    )

def test_bad_pred_proba(iris_labeled_dataset, iris_adaboost):
    # Arrange
    y_pred_train, _, y_proba_train, _ = _dummify_model(iris_labeled_dataset, None, iris_adaboost)

    y_proba_train = y_proba_train[:-1]

    # Act
    assert_that(
        calling(RocReport().run)
            .with_args(dataset=iris_labeled_dataset, y_pred=y_pred_train, y_proba=y_proba_train),
        raises(
            ValidationError,
            r'Prediction probabilities expected to be of length 150 but was: 149')
    )


def test_suite(diabetes_split_dataset_and_model):
    train, test, clf = diabetes_split_dataset_and_model
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = _dummify_model(train, test, clf)

    args = dict(train_dataset=train, test_dataset=test,
                y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                y_proba_train=y_proba_train, y_proba_test=y_proba_test)
    suite = full_suite()
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)


def test_predict_using_proba(iris_binary_string_split_dataset_and_model):
    train, test, clf = iris_binary_string_split_dataset_and_model
    y_pred_train, _, y_proba_train, _ = _dummify_model(train, test, clf)

    check = SingleDatasetPerformance(scorers=['f1_per_class'])

    proba_result = check.run(train, y_proba=y_proba_train)
    pred_result = check.run(train, y_pred=y_pred_train)

    assert_that(proba_result.value.iloc[0, -1], close_to(pred_result.value.iloc[0, -1], 0.001))
