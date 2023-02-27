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
"""Contains unit tests for the confusion_matrix_report check."""
from hamcrest import (assert_that, calling, close_to, greater_than, has_entries, has_entry, has_items, has_length, is_,
                      raises, equal_to)
from sklearn.metrics import f1_score, make_scorer, recall_score, get_scorer

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.model_evaluation import SimpleModelComparison
from deepchecks.tabular.metric_utils.scorers import get_default_scorers
from deepchecks.tabular.utils.task_type import TaskType
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(SimpleModelComparison().run).with_args(bad_dataset, bad_dataset, None),
                raises(DeepchecksValueError,
                       'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_classification_random(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='stratified')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, [0, 1, 2])


def test_classification_uniform(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='uniform')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, [0, 1, 2])


def test_classification_constant(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, [0, 1, 2])


def test_classification_binary_string_labels(iris_binary_string_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = iris_binary_string_split_dataset_and_model
    check = SimpleModelComparison()
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, ['a', 'b'])


def test_classification_binary_string_labels_custom_scorer(iris_binary_string_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = iris_binary_string_split_dataset_and_model
    check = SimpleModelComparison(scorers=[get_scorer('f1'), make_scorer(recall_score, average=None, zero_division=0)])
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result, equal_to({'scores': {'f1_score': {'Origin': 0.9411764705882353, 'Simple': 0.0},
                                             'recall_score': {'a': {'Origin': 0.9411764705882353, 'Simple': 1.0},
                                                              'b': {'Origin': 1.0, 'Simple': 0.0}}},
                                  'type': TaskType.BINARY,
                                  'scorers_perfect': {'f1_score': 1.0, 'recall_score': 1.0}}))


def test_classification_random_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='stratified',
                                  scorers={'recall': make_scorer(recall_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_classification(result.value, [0, 1, 2], ['recall'])
    assert_that(result.display, has_length(greater_than(0)))


def test_classification_random_custom_metric_without_display(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='stratified',
                                  scorers={'recall': make_scorer(recall_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf, with_display=False)
    # Assert
    assert_classification(result.value, [0, 1, 2], ['recall'])
    assert_that(result.display, has_length(0))


def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='stratified')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='stratified', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_uniform(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='uniform')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_condition_ratio_not_less_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison().add_condition_gain_greater_than(0.4)

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model is greater than 40%',
            details='Found failed metrics: {\'Neg RMSE\': \'24.32%\'}')
    ))


def test_condition_failed_for_multiclass(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(0.8)
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model is greater than 80%',
            details='Found classes with failed metric\'s gain: {1: {\'F1\': \'78.15%\'}}')
    ))


def test_condition_pass_for_multiclass_avg(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(0.43, average=True)
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            details='All metrics passed, metric\'s gain: {\'F1\': \'89.74%\'}',
            name='Model performance gain over simple model is greater than 43%')
    ))


def test_condition_pass_for_multiclass_avg_with_classes(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(1, average=False)\
        .add_condition_gain_greater_than(1, average=True, classes=[0])
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model is greater than 100%',
            details='Found classes with failed metric\'s gain: {1: {\'F1\': \'78.15%\'}, 2: {\'F1\': \'85.71%\'}}'
        ),
        equal_condition_result(
            is_pass=True,
            details='Found metrics with perfect score, no gain is calculated: [\'F1\']',
            name='Model performance gain over simple model is greater than 100% for classes [0]',
        )
    ))


def test_condition_pass_for_new_test_classes(kiss_dataset_and_model):
    train_ds, test_ds, clf = kiss_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent').add_condition_gain_greater_than(1)
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            details='Found metrics with perfect score, no gain is calculated: [\'F1\']',
            name='Model performance gain over simple model is greater than 100%',
        )
    ))


def test_condition_ratio_not_less_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison(strategy='stratified', n_samples=None).add_condition_gain_greater_than()

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            details='All metrics passed, metric\'s gain: {\'Neg RMSE\': \'52.17%\'}',
            name='Model performance gain over simple model is greater than 10%'
        )
    ))


def test_classification_tree(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, [0, 1, 2])


def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='tree',
                                  scorers={'recall': make_scorer(recall_score, average=None),
                                           'f1': make_scorer(f1_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, [0, 1, 2], ['recall', 'f1'])


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='most_frequent')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='tree', random_state=55)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree_max_depth(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(strategy='tree', max_depth=5)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def assert_regression(result):
    default_scorers = get_default_scorers(TaskType.REGRESSION)
    metric = next(iter(default_scorers))

    assert_that(result['scores'], has_entry(metric, has_entries({
        'Origin': close_to(-100, 100), 'Simple': close_to(-100, 100)
    })))
    assert_that(result['scorers_perfect'], has_entry(metric, is_(0)))


def assert_classification(result, classes, metrics=None):
    if not metrics:
        default_scorers = get_default_scorers(TaskType.MULTICLASS, class_avg=False)
        metrics = [next(iter(default_scorers))]
    class_matchers = {clas: has_entries({'Origin': close_to(1, 1), 'Simple': close_to(1, 1)})
                      for clas in classes}
    matchers = {metric: has_entries(class_matchers) for metric in metrics}
    assert_that(result['scores'], has_entries(matchers))
    assert_that(result['scorers_perfect'], has_entries({metric: is_(1) for metric in metrics}))
