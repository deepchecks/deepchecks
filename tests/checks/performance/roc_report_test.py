"""Contains unit tests for the roc_report check."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from hamcrest import assert_that, calling, raises, has_items

from deepchecks.base import Dataset
from deepchecks.checks.performance import RocReport
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(RocReport().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check RocReport requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(RocReport().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check RocReport requires dataset to have a label column'))


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(calling(RocReport().run).with_args(train, clf),
                raises(DeepchecksValueError, r'Check RocReport Expected model to be a type from'
                                           r' \[\'multiclass\', \'binary\'\], but received model of type: regression'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = RocReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result) == 3  # iris has 3 targets
    for value in result.values():
        assert isinstance(value, np.float64)


def test_condition_ratio_more_than_not_passed(iris_clean):
    # Arrange
    clf = LogisticRegression(max_iter=1)
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=55)
    clf.fit(x_train, y_train)
    ds = Dataset(pd.concat([x_test, y_test], axis=1), 
            features=iris_clean.feature_names,
            label='target')

    check = RocReport().add_condition_auc_not_less_than(min_auc=0.8)

    # Act
    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Not less than 0.8 AUC score for all the classes',
                               details='The scores that are less than the allowed AUC are: [\'class 1: 0.71\']')
    ))


def test_condition_ratio_more_than_passed(iris_clean):
    clf = LogisticRegression(max_iter=1)
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=55)
    clf.fit(x_train, y_train)
    ds = Dataset(pd.concat([x_test, y_test], axis=1), 
            features=iris_clean.feature_names,
            label='target')

    check = RocReport().add_condition_auc_not_less_than()

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Not less than 0.7 AUC score for all the classes',)
    )) 

    check = RocReport(excluded_classes=[1]).add_condition_auc_not_less_than(min_auc=0.8)

    result = check.conditions_decision(check.run(ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Not less than 0.8 AUC score for all the classes except: [1]',)
    )) 
