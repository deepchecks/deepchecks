"""The data_sample_leakage_report check module."""
from typing import Dict, List
import re

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

import pandas as pd
pd.options.mode.chained_assignment = None

__all__ = ['new_category', 'NewCategory']


def new_category(validation_dataset: Dataset, train_dataset: Dataset):
    """Find new category in validation.

    Args:
        train_dataset (Dataset): The training dataset object.
        validation_dataset (Dataset): The validation dataset object.
    Returns:
        CheckResult: value is sample leakage ratio in %,
                     displays a dataframe that shows the duplicated rows between the datasets

    Raises:
        MLChecksValueError: If the object is not a Dataset instance

    """
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    result = None
    display = None

    return CheckResult(result, check=new_category, display=display)

class NewCategory(TrainValidationBaseCheck):
    """Find new category in validation."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Run new_category check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
        Returns:
            CheckResult: value is sample leakage ratio in %,
                         displays a dataframe that shows the duplicated rows between the datasets
        """
        return new_category(validation_dataset=validation_dataset, train_dataset=train_dataset)
