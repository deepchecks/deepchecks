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
from deepchecks.utils.strings import format_number
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result

from hamcrest import assert_that, calling, raises, close_to, has_items, has_length


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
    # Assert - 3 classes X 1 metrics X 2 models
    assert_that(result['scores'], has_length(6))


def test_classification_constant(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 3 classes X 1 metrics X 2 models
    assert_that(result['scores'], has_length(6))


def test_classification_random_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random',
                                  alternative_scorers={'recall': make_scorer(recall_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 3 classes X 1 metrics X 2 models
    assert_that(result['scores'], has_length(6))


def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


def test_regression_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


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
            name='Ratio not less than 1.4 between the given model\'s score and the simple model\'s score',
            details='Metrics with scores ratio lower than threshold: RMSE - Default')
    ))


def test_condition_ratio_not_less_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison(simple_model_type='random').add_condition_ratio_not_less_than(min_allowed_ratio=1.1)

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check_result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='Ratio not less than 1.1 between the given model\'s score and the simple model\'s score'
        )
    ))


def test_classification_tree(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 3 classes X 1 metrics X 2 models
    assert_that(result['scores'], has_length(6))


def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree',
                                  alternative_scorers={'recall': make_scorer(recall_score, average=None),
                                                       'f1': make_scorer(f1_score, average=None)})
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 3 classes X 2 metrics X 2 models
    assert_that(result['scores'], has_length(12))


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


def test_regression_tree(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


def test_regression_tree_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', random_state=55)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))


def test_regression_tree_max_depth(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', max_depth=5)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert - 1 metrics X 2 models
    assert_that(result['scores'], has_length(2))
