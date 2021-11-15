"""
Contains unit tests for the data_sample_leakage_report check
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.base import Dataset
from deepchecks.utils import DeepchecksValueError
from deepchecks.checks.leakage import DataSampleLeakageReport
from hamcrest import assert_that, calling, raises, equal_to


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(DataSampleLeakageReport().run).with_args(x, x),
                raises(DeepchecksValueError,
                'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_no_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                features=iris_clean.feature_names,
                label='target')

    test_df = pd.concat([x_test, y_test], axis=1)

    test_dataset = Dataset(test_df,
                features=iris_clean.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, equal_to(0))

def test_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                features=iris_clean.feature_names,
                label='target')

    test_df = pd.concat([x_test, y_test], axis=1)
    bad_test = test_df.append(train_dataset.data.iloc[[0, 1, 2, 3, 4]], ignore_index=True)

    test_dataset = Dataset(bad_test,
                features=iris_clean.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, equal_to(0.1))


def test_nan():
    train_dataset = Dataset(pd.DataFrame({'col1': [1, 2, 3, np.nan], 'col2': [1, 2, 1, 1]}),
                            label='col2')
    test_dataset = Dataset(pd.DataFrame({'col1': [2, np.nan, np.nan, np.nan], 'col2': [1, 1, 2, 1]}),
                                 label='col2')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(test_dataset=test_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, equal_to(0.5))
