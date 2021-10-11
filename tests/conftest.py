import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

# pylint: disable=redefined-outer-name
# Conftest using pytest fixtures to define parameters with
# names of functions as the desired behaviour

@pytest.fixture(scope='session')
def iris():
    return load_iris()


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    x = iris.data
    y = iris.target
    clf.fit(x, y)
    return clf
