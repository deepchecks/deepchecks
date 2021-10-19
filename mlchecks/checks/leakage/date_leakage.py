"""The date_leakage check module."""
import pandas as pd

from mlchecks import CheckResult, Dataset, TrainValidationBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.display import format_check_display

__all__ = ['validation_dates_before_train_leakage',
           'ValidationDatesBeforeTrainLeakage',
           'date_train_validation_leakage',
           'DateTrainValidationLeakage']

def validation_dates_before_train_leakage(train_dataset: Dataset, validation_dataset: Dataset):
    """
    Check the percentage of rows in validation that are dated earlier than max date in train.

    Args:
       train_dataset (Dataset): The training dataset object. Must contain a date column.
       validation_dataset (Dataset): The validation dataset object. Must contain a date column.

    Returns:
       CheckResult:
            - value is the ratio of date leakage.
            - data is html display of the checks' textual result.

    Raises:
        MLChecksValueError: If one of the datasets is not a Dataset instance with an date
    """
    train_dataset = validate_dataset(train_dataset, validation_dates_before_train_leakage.__name__)
    validation_dataset = validate_dataset(validation_dataset, validation_dates_before_train_leakage.__name__)
    train_dataset.validate_date(validation_dates_before_train_leakage.__name__)
    validation_dataset.validate_date(validation_dates_before_train_leakage.__name__)

    train_date = train_dataset.date_col()
    val_date = validation_dataset.date_col()

    max_train_date = max(train_date)
    dates_leaked = sum(date <= max_train_date for date in val_date)

    if dates_leaked > 0:
        size_in_test = dates_leaked / validation_dataset.n_samples()
        text = f'{size_in_test:.1%} of validation data dates before last training data date ({max_train_date})'
        display_str = f'{text}'
        return_value = size_in_test
    else:
        display_str = None
        return_value = 0

    return CheckResult(value=return_value,
                       display={'text/html':
                                format_check_display('Date Train-Validation Leakage',
                                                     validation_dates_before_train_leakage,
                                                     display_str)})


class ValidationDatesBeforeTrainLeakage(TrainValidationBaseCheck):
    """Check validation data that is dated earlier than latest date in train."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the date_leakage check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date column.
            validation_dataset (Dataset): The validation dataset object. Must contain an date column.
            model: any = None - not used in the check

        Returns:
            the output of the date_leakage check
        """
        return validation_dates_before_train_leakage(train_dataset, validation_dataset)


def date_train_validation_leakage(train_dataset: Dataset, validation_dataset: Dataset, n_dates_to_show: int = 5):
    """
    Check if validation dates are present in train data.

    Args:
       train_dataset (Dataset): The training dataset object. Must contain an date.
       validation_dataset (Dataset): The validation dataset object. Must contain an date.
       n_dates_to_show (int): Number of common dates to show.

    Returns:
       CheckResult:
            - value is the ratio of date leakage.
            - data is html display of the checks' textual result.

    Raises:
        MLChecksValueError: If one of the datasets is not a Dataset instance with an date
    """
    train_dataset = validate_dataset(train_dataset, date_train_validation_leakage.__name__)
    validation_dataset = validate_dataset(validation_dataset, date_train_validation_leakage.__name__)
    train_dataset.validate_date(date_train_validation_leakage.__name__)
    validation_dataset.validate_date(date_train_validation_leakage.__name__)

    train_index = train_dataset.date_col()
    val_index = validation_dataset.date_col()

    date_intersection = list(set(train_index).intersection(val_index))
    if len(date_intersection) > 0:
        size_in_test = len(date_intersection) / validation_dataset.n_samples()
        text = f'{size_in_test:.1%} of validation data dates appear in training data'
        table = pd.DataFrame([[list(date_intersection)[:n_dates_to_show]]],
                             index=['Sample of validation dates in train:'])
        display_str = f'{text}<br>{table.to_html(header=False)}'
        return_value = size_in_test
    else:
        display_str = None
        return_value = 0

    return CheckResult(value=return_value,
                       display={'text/html':
                                format_check_display('Index Train-Validation Leakage', date_train_validation_leakage,
                                                     display_str)})


class DateTrainValidationLeakage(TrainValidationBaseCheck):
    """Check if validation dates are present in train data."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the date_train_validation_leakage check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date.
            validation_dataset (Dataset): The validation dataset object. Must contain an date.
            model: any = None - not used in the check

        Returns:
            the output of the index_train_validation_leakage check
        """
        return date_train_validation_leakage(train_dataset, validation_dataset,
                                              n_dates_to_show=self.params.get('n_dates_to_show'))
