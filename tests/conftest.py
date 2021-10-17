"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name

import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
from mlchecks import Dataset


@pytest.fixture(scope='session')
def iris():
    """Return Iris dataset as DataFrame."""
    df = load_iris(return_X_y=False, as_frame=True)
    return pd.concat([df.data, df.target], axis=1)


@pytest.fixture(scope='session')
def iris_dataset(iris):
    """Return Iris dataset as Dataset object."""
    return Dataset(iris)


@pytest.fixture(scope='session')
def iris_dataset_labeled(iris):
    """Return Iris dataset as Dataset object."""
    return Dataset(iris, label='target')


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    """Return trained AdaBoostClassifier on iris data."""
    clf = AdaBoostClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf


@pytest.fixture(scope='session')
def iris_labled_dataset(iris):
    """Return t Iris dataset as Dataset object with label."""
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