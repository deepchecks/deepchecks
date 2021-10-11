import pytest
import joblib

@pytest.fixture(scope='session')
def skmodel():
    return joblib.load('tests/assets/model.joblib')
