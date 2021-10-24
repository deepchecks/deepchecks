"""module contains Data Duplicates check."""
from typing import Iterable, Union

import pandas as pd

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck, TrainValidationBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.utils import MLChecksValueError

__all__ = ['data_duplicates', 'DataDuplicates']


def data_sample_leakage_report(validation_dataset: Dataset, train_dataset: Dataset, thres: float = 0.7):
    """Find which percent of the validation data in the train data.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain an index.
        validation_dataset (Dataset): The validation dataset object. Must contain an index.
    Returns:
        CheckResult: value is sample leakage ratio in %,
                     displays a dataframe that shows the duplicated rows between the datasets

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_features(train_dataset, data_sample_leakage_report.__name__)

    columns = train_dataset.features()

    train_f = train_dataset.data
    val_f = validation_dataset.data

    for column in columns:
        top_val10 = val_f[column].value_counts().head(10)
        top_train10 = train_f[column].value_counts().head(10)




    
    return CheckResult(dup_ratio, header='Data Sample Leakage Report',
                       check=data_sample_leakage_report, display=display)

class DataSampleLeakageReport(TrainValidationBaseCheck):
    """Finds data sample leakage."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Run data_sample_leakage_report check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
        Returns:
            CheckResult: value is sample leakage ratio in %,
                         displays a dataframe that shows the duplicated rows between the datasets
        """
        return data_sample_leakage_report(validation_dataset=validation_dataset, train_dataset=train_dataset)
