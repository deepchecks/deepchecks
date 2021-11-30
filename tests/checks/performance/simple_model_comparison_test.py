"""Contains unit tests for the confusion_matrix_report check."""
from deepchecks.checks.performance import SimpleModelComparison
from deepchecks.utils.strings import format_number
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result

from hamcrest import assert_that, calling, raises, close_to, has_items


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(SimpleModelComparison().run).with_args(bad_dataset, bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check SimpleModelComparison requires dataset to be of type Dataset. instead got: str'))


def test_classification_random(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['simple_model_score'], close_to(0.2, 0.5))


def test_classification_constant(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['simple_model_score'], close_to(0.3, 0.5))


def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random', metric='recall_micro')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['simple_model_score'], close_to(0.2, 0.5))


def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(-57, 0.5))
    assert_that(result['simple_model_score'], close_to(-114, 0.5))

def test_regression_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='random', random_state=0)
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(-57, 0.5))
    assert_that(result['simple_model_score'], close_to(-105, 0.5))


def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    result = check.run(train_ds, test_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(-57, 0.5))
    assert_that(result['simple_model_score'], close_to(-76, 0.5))


def test_condition_ratio_not_less_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison().add_condition_ratio_not_less_than(min_allowed_ratio=1.4)

    # Act
    check_result = check.run(train_ds, test_ds, clf)
    condition_result = check.conditions_decision(check_result)
    ratio = check_result.value['ratio']

    assert_that(ratio, close_to(1.32, 0.03))
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               name='Ratio not less than 1.4 '
                                    'between the given model\'s result and the simple model\'s result',
                               details=f'The given model performs {format_number(ratio)} times compared' \
                                       'to the simple model using the given metric')
    ))


def test_condition_ratio_not_less_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    check = SimpleModelComparison().add_condition_ratio_not_less_than(min_allowed_ratio=1.3)

    # Act
    result = check.conditions_decision(check.run(train_ds, test_ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ratio not less than 1.3 '
                                    'between the given model\'s result and the simple model\'s result')
    ))

def test_classification_tree(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(0.95, 0.05))

def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, test_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', metric='recall_micro')
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(0.95, 0.05))

def test_regression_constant(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='constant')
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(1.32, 0.05))

def test_regression_tree(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree')
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(1, 0.09))

def test_regression_tree_random_state(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', random_state=55)
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(1, 0.09))

def test_regression_tree_max_depth(diabetes_split_dataset_and_model):
    train_ds, test_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = SimpleModelComparison(simple_model_type='tree', max_depth=5)
    # Act X
    ratio = check.run(train_ds, test_ds, clf).value['ratio']
    # Assert
    assert_that(ratio, close_to(1.1, 0.08))
