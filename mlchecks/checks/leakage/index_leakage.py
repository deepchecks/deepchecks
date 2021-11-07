"""The index_leakage check module."""
import pandas as pd

from mlchecks import CheckResult, Dataset, TrainValidationBaseCheck

__all__ = ['IndexTrainValidationLeakage']


class IndexTrainValidationLeakage(TrainValidationBaseCheck):
    """Check if validation indexes are present in train data."""

    def __init__(self, n_index_to_show: int = 5):
        """Initialize the IndexTrainValidationLeakage check.

        Args:
            n_index_to_show (int): Number of common indexes to show.
        """
        super().__init__()
        self.n_index_to_show = n_index_to_show

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
            model: any = None - not used in the check

        Returns:
           CheckResult:
                - value is the ratio of index leakage.
                - data is html display of the checks' textual result.

        Raises:
            MLChecksValueError: If the if one of the datasets is not a Dataset instance with an index
        """
        return self._index_train_validation_leakage(train_dataset, validation_dataset)

    def _index_train_validation_leakage(self, train_dataset: Dataset, validation_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset, self.__class__.__name__)
        validation_dataset = Dataset.validate_dataset(validation_dataset, self.__class__.__name__)
        train_dataset.validate_index(self.__class__.__name__)
        validation_dataset.validate_index(self.__class__.__name__)

        train_index = train_dataset.index_col()
        val_index = validation_dataset.index_col()

        index_intersection = list(set(train_index).intersection(val_index))
        if len(index_intersection) > 0:
            size_in_test = len(index_intersection) / validation_dataset.n_samples()
            text = f'{size_in_test:.1%} of validation data indexes appear in training data'
            table = pd.DataFrame([[list(index_intersection)[:self.n_index_to_show]]],
                                 index=['Sample of validation indexes in train:'])
            display = [text, table]
        else:
            size_in_test = 0
            display = None

        return CheckResult(value=size_in_test, header='Index Train-Validation Leakage', check=self.__class__,
                           display=display)
