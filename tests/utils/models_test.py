import numpy as np
import pandas as pd

from deepchecks.utils.models import RandomModel, PerfectModel
from hamcrest import assert_that, contains_exactly, is_


def test_random_model():
    # Arrange
    model = RandomModel()
    y = pd.Series([1, 2, 1, 2, 3])
    x = np.ones(5)
    model.fit([], y)
    # Act
    np.random.seed(42)
    p = model.predict(x)
    np.random.seed(42)
    proba = model.predict_proba(x)
    # Assert
    assert_that(p.tolist(), contains_exactly(2, 3, 1, 3, 3))
    expected_proba = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0],
                              [0, 0, 1],
                              [0, 0, 1]])
    assert_that(np.equal(proba, expected_proba).sum(), is_(15))


def test_perfect_model():
    # Arrange
    model = PerfectModel()
    data = pd.DataFrame(data={'target': [1, 2, 1, 2, 3], 'col1': ['a', 'b', 'a', 'a', 'c']})
    model.fit([], data['target'])
    # Act
    p = model.predict(data[['col1']])
    proba = model.predict_proba(data[['col1']])
    # Assert
    assert_that(p.tolist(), contains_exactly(1, 2, 1, 2, 3))
    expected_proba = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    assert_that(np.equal(proba, expected_proba).sum(), is_(15))
