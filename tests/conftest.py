"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name
from typing import Tuple

import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

from mlchecks import Dataset


@pytest.fixture(scope='session')
def empty_df():
    return pd.DataFrame([])


@pytest.fixture(scope='session')
def diabetes():
    """Return diabetes dataset splited to train and validation as Datasets."""
    diabetes = load_diabetes(return_X_y=False, as_frame=True).frame
    train_df, validation_df = train_test_split(diabetes, test_size=0.33)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')
    return train, validation


@pytest.fixture(scope='session')
def diabetes_model(diabetes):
    clf = GradientBoostingRegressor()
    train, _ = diabetes
    return clf.fit(train.features_columns(), train.label_col())


@pytest.fixture(scope='session')
def diabetes_split_dataset_and_model(diabetes, diabetes_model):
    train, validation = diabetes
    clf = diabetes_model
    return train, validation, clf


@pytest.fixture(scope='session')
def iris_clean():
    """Return Iris dataset as DataFrame."""
    iris = load_iris(return_X_y=False, as_frame=True)
    return iris


@pytest.fixture(scope='session')
def iris(iris_clean):
    """Return Iris dataset as DataFrame."""
    return iris_clean.frame


@pytest.fixture(scope='session')
def iris_dataset(iris):
    """Return Iris dataset as Dataset object."""
    return Dataset(iris)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    """Return trained AdaBoostClassifier on iris data."""
    clf = AdaBoostClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_labeled_dataset(iris):
    """Return Iris dataset as Dataset object with label."""
    return Dataset(iris, label='target')


@pytest.fixture(scope='session')
def iris_random_forest(iris):
    """Return trained RandomForestClassifier on iris data."""
    clf = RandomForestClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_random_forest_single_class(iris):
    """Return trained RandomForestClassifier on iris data modified to a binary label."""
    clf = RandomForestClassifier()
    idx = iris.target != 2
    features = iris.drop('target', axis=1)[idx]
    target = iris.target[idx]
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_dataset_single_class(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df)
    return dataset


@pytest.fixture(scope='session')
def iris_dataset_single_class_labeled(iris):
    """Return Iris dataset modified to a binary label as Dataset object."""
    idx = iris.target != 2
    df = iris[idx]
    dataset = Dataset(df, label='target')
    return dataset


@pytest.fixture(scope='session')
def iris_split_dataset_and_model(iris) -> Tuple[Dataset, Dataset, AdaBoostClassifier]:
    """Return Iris train and val datasets and trained RF model."""
    iris = load_iris(as_frame=True)
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)
    train_ds = Dataset(pd.concat([x_train, y_train], axis=1),
                       features=iris.feature_names,
                       label='target')
    val_ds = Dataset(pd.concat([x_test, y_test], axis=1),
                     features=iris.feature_names,
                     label='target')
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    return train_ds, val_ds, clf
