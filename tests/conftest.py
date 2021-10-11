import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

@pytest.fixture(scope='session')
def iris_adaboost():
    clf = AdaBoostClassifier()
    iris = load_iris()
    X = iris.data
    Y = iris.target
    clf.fit(X, Y)
    return clf
