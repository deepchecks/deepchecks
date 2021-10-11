import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris


@pytest.fixture(scope='session')
def iris():
    return load_iris()


@pytest.fixture(scope='session')
def iris_adaboost(iris):
    clf = AdaBoostClassifier()
    X = iris.data
    Y = iris.target
    clf.fit(X, Y)
    return clf
