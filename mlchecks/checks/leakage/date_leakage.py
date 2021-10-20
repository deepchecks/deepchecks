"""The date_leakage check module."""
import pandas as pd

from mlchecks import CheckResult, Dataset, TrainValidationBaseCheck

__all__ = ['date_train_validation_leakage_overlap',
           'DateTrainValidationLeakageOverlap',
           'date_train_validation_leakage_duplicates',
           'DateTrainValidationLeakageDuplicates']

def date_train_validation_leakage_overlap(train_dataset: Dataset, validation_dataset: Dataset):
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
    self = date_train_validation_leakage_overlap
    train_dataset = Dataset.validate_dataset(train_dataset, self.__name__)
    validation_dataset = Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_date(self.__name__)
    validation_dataset.validate_date(self.__name__)

    train_date = train_dataset.date_col()
    val_date = validation_dataset.date_col()

    max_train_date = max(train_date)
    dates_leaked = sum(date <= max_train_date for date in val_date)

    if dates_leaked > 0:
        size_in_test = dates_leaked / validation_dataset.n_samples()
        display = f'{size_in_test:.1%} of validation data dates before last training data date ({max_train_date})'
        return_value = size_in_test
    else:
        display = None
        return_value = 0

    return CheckResult(value=return_value,
                       header='Date Train-Validation Leakage (overlap)',
                       check=self,
                       display=display)


class DateTrainValidationLeakageOverlap(TrainValidationBaseCheck):
    """Check validation data that is dated earlier than latest date in train."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the date_train_validation_leakage_overlap check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date column.
            validation_dataset (Dataset): The validation dataset object. Must contain an date column.
            model: any = None - not used in the check

        Returns:
            the output of the date_train_validation_leakage_overlap check
        """
        return date_train_validation_leakage_overlap(train_dataset, validation_dataset)


def date_train_validation_leakage_duplicates(train_dataset: Dataset, validation_dataset: Dataset, n_to_show: int = 5):
    """
    Check if validation dates are present in train data.

    Args:
       train_dataset (Dataset): The training dataset object. Must contain an date.
       validation_dataset (Dataset): The validation dataset object. Must contain an date.
       n_to_show (int): Number of common dates to show.

    Returns:
       CheckResult:
            - value is the ratio of date leakage.
            - data is html display of the checks' textual result.

    Raises:
        MLChecksValueError: If one of the datasets is not a Dataset instance with an date
    """
    self = date_train_validation_leakage_duplicates
    train_dataset = Dataset.validate_dataset(train_dataset, self.__name__)
    validation_dataset = Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_date(self.__name__)
    validation_dataset.validate_date(self.__name__)

    train_date = train_dataset.date_col()
    val_date = validation_dataset.date_col()

    date_intersection = set(train_date).intersection(val_date)
    if len(date_intersection) > 0:
        size_in_test = len(date_intersection) / validation_dataset.n_samples()
        text = f'{size_in_test:.1%} of validation data dates appear in training data'
        table = pd.DataFrame([[list(date_intersection)[:n_to_show]]],
                             index=['Sample of validation dates in train:'])
        display = [text, table]
        return_value = size_in_test
    else:
        display = None
        return_value = 0

    return CheckResult(value=return_value,
                       header='Date Train-Validation Leakage (duplicates)',
                       check=self,
                       display=display)


class DateTrainValidationLeakageDuplicates(TrainValidationBaseCheck):
    """Check if validation dates are present in train data."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the date_train_validation_leakage_duplicates check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date.
            validation_dataset (Dataset): The validation dataset object. Must contain an date.
            model: any = None - not used in the check

        Returns:
            the output of the date_train_validation_leakage_duplicates check
        """
        return date_train_validation_leakage_duplicates(train_dataset, validation_dataset,
                                                        n_to_show=self.params.get('n_to_show'))
