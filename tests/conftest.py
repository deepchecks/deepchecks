"""Represents fixtures for unit testing using pytest."""
# Disable this pylint check since we use this convention in pytest fixtures
#pylint: disable=redefined-outer-name

import pytest
from sklearn.ensemble import AdaBoostClassifier
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
