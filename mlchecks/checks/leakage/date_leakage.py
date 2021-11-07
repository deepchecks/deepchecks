"""The date_leakage check module."""
import pandas as pd

from mlchecks import CheckResult, Dataset, TrainValidationBaseCheck

__all__ = ['DateTrainValidationLeakageOverlap',
           'DateTrainValidationLeakageDuplicates']


class DateTrainValidationLeakageOverlap(TrainValidationBaseCheck):
    """Check validation data that is dated earlier than latest date in train."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date column.
            validation_dataset (Dataset): The validation dataset object. Must contain an date column.
            model: any = None - not used in the check

        Returns:
           CheckResult:
                - value is the ratio of date leakage.
                - data is html display of the checks' textual result.

        Raises:
            MLChecksValueError: If one of the datasets is not a Dataset instance with an date
        """
        return self._date_train_validation_leakage_overlap(train_dataset, validation_dataset)

    def _date_train_validation_leakage_overlap(self, train_dataset: Dataset, validation_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset, self._date_train_validation_leakage_overlap.__name__)
        validation_dataset = Dataset.validate_dataset(validation_dataset,
                                                      self._date_train_validation_leakage_overlap.__name__)
        train_dataset.validate_date(self._date_train_validation_leakage_overlap.__name__)
        validation_dataset.validate_date(self._date_train_validation_leakage_overlap.__name__)

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
                           check=self.__class__,
                           display=display)


class DateTrainValidationLeakageDuplicates(TrainValidationBaseCheck):
    """Check if validation dates are present in train data."""

    def __init__(self, n_to_show: int = 5):
        """Initialize the DateTrainValidationLeakageDuplicates check.

        Args:
            n_to_show (int): Number of common dates to show.
        """
        super().__init__()
        self.n_to_show = n_to_show

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an date.
            validation_dataset (Dataset): The validation dataset object. Must contain an date.
            model: any = None - not used in the check

        Returns:
           CheckResult:
                - value is the ratio of date leakage.
                - data is html display of the checks' textual result.

        Raises:
            MLChecksValueError: If one of the datasets is not a Dataset instance with an date
        """
        return self._date_train_validation_leakage_duplicates(train_dataset, validation_dataset)

    def _date_train_validation_leakage_duplicates(self, train_dataset: Dataset, validation_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset, self.__class__.__name__)
        validation_dataset = Dataset.validate_dataset(validation_dataset, self.__class__.__name__)
        train_dataset.validate_date(self.__class__.__name__)
        validation_dataset.validate_date(self.__class__.__name__)

        train_date = train_dataset.date_col()
        val_date = validation_dataset.date_col()

        date_intersection = set(train_date).intersection(val_date)
        if len(date_intersection) > 0:
            size_in_test = len(date_intersection) / validation_dataset.n_samples()
            text = f'{size_in_test:.1%} of validation data dates appear in training data'
            table = pd.DataFrame([[list(date_intersection)[:self.n_to_show]]],
                                 index=['Sample of validation dates in train:'])
            display = [text, table]
            return_value = size_in_test
        else:
            display = None
            return_value = 0

        return CheckResult(value=return_value, header='Date Train-Validation Leakage (duplicates)',
                           check=self.__class__, display=display)
