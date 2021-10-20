import pandas as pd
from sklearn.model_selection import train_test_split
from mlchecks.base import Dataset
from mlchecks.checks.leakage import DataSampleLeakageReport, data_sample_leakage_report
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(data_sample_leakage_report).with_args(X, X),
                raises(MLChecksValueError, 'function data_sample_leakage requires dataset to be of type Dataset. instead got: str'))


def test_no_leakage(iris):
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), 
                features=iris.feature_names,
                label='target')

    test_df = pd.concat([X_test, y_test], axis=1)
                        
    validation_dataset = Dataset(test_df, 
                features=iris.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert(result == 0)

def test_leakage(iris):
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
    train_dataset = Dataset(pd.concat([X_train, y_train], axis=1), 
                features=iris.feature_names,
                label='target')

    test_df = pd.concat([X_test, y_test], axis=1)
    bad_test = test_df.append(train_dataset.iloc[[0, 1, 2, 3, 4]], ignore_index=True)
                        
    validation_dataset = Dataset(bad_test, 
                features=iris.feature_names,
                label='target')
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert(result == 12)
