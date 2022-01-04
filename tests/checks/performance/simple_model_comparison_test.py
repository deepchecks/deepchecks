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
"""Contains unit tests for the confusion_matrix_report check."""
from sklearn.metrics import make_scorer, recall_score, f1_score

from deepchecks.checks.performance import SimpleModelComparison
from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.metrics import DEFAULT_SINGLE_SCORER, ModelType, DEFAULT_SINGLE_SCORER_MULTICLASS_NON_AVG
from tests.checks.utils import equal_condition_result

from hamcrest import assert_that, calling, raises, close_to, has_items, has_entries, has_entry, is_, contains_exactly


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(SimpleModelComparison().run).with_args(bad_dataset, bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_classification_random(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result)


def test_classification_constant(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result)


def test_classification_random_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random',
                                  alternative_scorers={'recall': make_scorer(recall_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, ['recall'])


def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_condition_ratio_not_less_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison().add_condition_ratio_not_less_than(min_allowed_ratio=1.4)

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=False,
            name='$$\\frac{\\text{model score}}{\\text{simple model score}} >= 1.4$$',
            details='Metrics failed: "Neg RMSE (Default)"')
    ))


def test_condition_failed_for_multiclass(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant').add_condition_gain_not_less_than(0.8)
    # Act X
    result = check.run(train_ds, test_ds, clf)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=False,
            name='Model performance gain over simple model must be at least 80.00%',
            details='Metrics failed: "F1 (Default)" - Classes: 1')
    ))


def test_condition_ratio_not_less_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison(simple_model_type='random').add_condition_gain_not_less_than()

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='Model performance gain over simple model must be at least 10.00%'
        )
    ))


def test_classification_tree(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result)


def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree',
                                  alternative_scorers={'recall': make_scorer(recall_score, average=None),
                                                       'f1': make_scorer(f1_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_classification(result, ['recall', 'f1'])


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', random_state=55)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def test_regression_tree_max_depth(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', max_depth=5)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_regression(result)


def assert_regression(result):
    metric = DEFAULT_SINGLE_SCORER[ModelType.REGRESSION]
    assert_that(result['scores'], has_entry(metric, has_entries({
        'Origin': close_to(-100, 100), 'Simple': close_to(-100, 100)
    })))
    assert_that(result['scorers_perfect'], has_entry(metric, is_(0)))
    assert_that(result['classes'], is_(None))


def assert_classification(result, metrics=None):
    metrics = metrics or [DEFAULT_SINGLE_SCORER_MULTICLASS_NON_AVG]
    class_matchers = {clas: has_entries({'Origin': close_to(1, 1), 'Simple': close_to(1, 1)})
                      for clas in result['classes']}
    matchers = {metric: has_entries(class_matchers) for metric in metrics}
    assert_that(result['scores'], has_entries(matchers))
    assert_that(result['scorers_perfect'], has_entries({metric: is_(1) for metric in metrics}))
    assert_that(result['classes'], contains_exactly(0, 1, 2))
