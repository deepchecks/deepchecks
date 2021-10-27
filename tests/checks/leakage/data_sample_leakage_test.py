"""
Contains unit tests for the data_sample_leakage_report check
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from mlchecks.base import Dataset
from mlchecks.utils import MLChecksValueError
from mlchecks.checks.leakage import DataSampleLeakageReport, data_sample_leakage_report
from hamcrest import assert_that, calling, raises, equal_to


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(data_sample_leakage_report).with_args(x, x),
                raises(MLChecksValueError,
                'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_no_leakage(iris_clean):
    x = iris_clean.data
    y = iris_clean.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([x_train, y_train], axis=1),
                features=iris_clean.feature_names,
                label='target')

    test_df = pd.concat([x_test, y_test], axis=1)

    validation_dataset = Dataset(test_df,
                features=iris_clean.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
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

    validation_dataset = Dataset(bad_test,
                features=iris_clean.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, equal_to(0.12))
