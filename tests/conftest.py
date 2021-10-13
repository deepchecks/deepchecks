"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name

import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
from mlchecks.base import Dataset


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
def iris_train_val_ds(iris):
    """Return Iris train and validation as Dataset object."""
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), 
                features=iris.feature_names,
                label='target')

    validation_dataset = Dataset(pd.concat([X_test, y_test], axis=1),                    
                features=iris.feature_names,
                label='target')
    
    return (validation_dataset, train_dataset)


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
