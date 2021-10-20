from mlchecks.checks.leakage import DataSampleLeakageReport, data_sample_leakage_report
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(data_sample_leakage_report).with_args(X, X),
                raises(MLChecksValueError, 'function data_sample_leakage requires dataset to be of type Dataset. instead got: str'))


def test_model_info_object(iris_train_val_ds):
    (validation_dataset, train_dataset) = iris_train_val_ds
    # Arrange
    check = DataSampleLeakageReport()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert(result == 12) # iris has 3 targets
