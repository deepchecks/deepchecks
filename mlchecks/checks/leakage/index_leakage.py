"""The index_leakage check module."""
import pandas as pd

from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck, TrainValidationBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.display import format_check_display
from mlchecks.utils import is_notebook, MLChecksValueError, get_plt_html_str

__all__ = ['index_train_validation_leakage', 'IndexTrainValidationLeakage']


def index_train_validation_leakage(train_dataset: Dataset, validation_dataset: Dataset, n_index_to_show: int = 5):
    """
    Check if validation indexes are present in train data.

    Args:
       train_dataset (Dataset): The training dataset object. Must contain an index.
       validation_dataset (Dataset): The validation dataset object. Must contain an index.
       n_index_to_show (int): Number of common indexes to show.

    Returns:
       CheckResult:
            - value is the ratio of index leakage.
            - data is html display of the checks' textual result.

    Raises:
        MLChecksValueError: If the if one of the datasets is not a Dataset instance with an index
    """
    train_dataset = validate_dataset(train_dataset, index_train_validation_leakage.__name__)
    validation_dataset = validate_dataset(validation_dataset, index_train_validation_leakage.__name__)
    train_dataset.validate_index(index_train_validation_leakage.__name__)
    validation_dataset.validate_index(index_train_validation_leakage.__name__)

    train_index = train_dataset.index_col()
    val_index = validation_dataset.index_col()

    index_intersection = list(set(train_index).intersection(val_index))
    if len(index_intersection) > 0:
        size_in_test = len(index_intersection) / validation_dataset.n_samples()
        text = f'{size_in_test:.1%} of validation data indexes appear in training data'
        table = pd.DataFrame([[list(index_intersection)[:n_index_to_show]]],
                             index=['Sample of validation indexes in train:'])
        display_str = f'{text}<br>{table.to_html(header=False)}'
        return_value = size_in_test
    else:
        display_str = None
        return_value = 0

    return CheckResult(value=return_value,
                       display={'text/html':
                                format_check_display('Index Train-Validation Leakage', index_train_validation_leakage,
                                                     display_str)})


class IndexTrainValidationLeakage(TrainValidationBaseCheck):
    """Check if validation indexes are present in train data."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """
        Run the index_train_validation_leakage check.

        Arguments:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
            model: any = None - not used in the check

        Returns:
            the output of the index_train_validation_leakage check
        """
        return index_train_validation_leakage(train_dataset, validation_dataset,
                                              n_index_to_show=self.params.get('n_index_to_show'))
