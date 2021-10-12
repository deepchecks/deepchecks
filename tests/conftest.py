"""Represents fixtures for unit testing using pytest."""
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
import pandas as pd
from mlchecks import Dataset


@pytest.fixture(scope='session')
def iris():
    df = load_iris(return_X_y=False, as_frame=True)
    return pd.concat([df.data, df.target], axis=1)


@pytest.fixture(scope='session')
def iris_dataset(iris):
    return Dataset(iris)


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    features = iris.drop('target', axis=1)
    target = iris.target
    clf.fit(features, target)
    return clf
