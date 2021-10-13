"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name

import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
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
