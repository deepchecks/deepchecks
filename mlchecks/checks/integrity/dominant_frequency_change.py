"""module contains Dominant Frequency Change check."""
from typing import  Dict

from scipy.stats import chi2_contingency, fisher_exact
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, TrainValidationBaseCheck

__all__ = ['dominant_frequency_change', 'DominantFrequencyChange']


def find_p_val(key: str, ref_hist: Dict, test_hist: Dict, ref_count: int, test_count: int, ratio_change_thres: float) -> float:
    """find p value for column frequency change between the reference dataset to the test dataset

    Args:
        key (str): key of the dominant value.
        ref_hist (Dict): The reference dataset histogram.
        test_hist (Dict): The test dataset histogram.
        ref_count (int): The reference dataset row count.
        test_count (int): The test dataset row count.
        ratio_change_thres (float): The dominant frequency has to change by at least this ratio (0-inf).
    Returns:
        float: p value for the key.

    Raises:
        MLChecksValueError: If the object is not a Dataset or DataFrame instance

    """
    contingency_matrix_df = pd.DataFrame(np.zeros((2, 2)), index=["dominant", "others"], columns=["ref", "test"])
    contingency_matrix_df.loc["dominant", "ref"] = ref_hist.get(key, 0)
    contingency_matrix_df.loc["dominant", "test"] = test_hist.get(key, 0)
    contingency_matrix_df.loc["others", "ref"] = ref_count - ref_hist.get(key, 0)
    contingency_matrix_df.loc["others", "test"] = test_count - test_hist.get(key, 0)

    test_percent = contingency_matrix_df.loc["dominant", "test"] / test_count
    ref_percent = contingency_matrix_df.loc["dominant", "ref"] / ref_count
    if ref_percent == 0 or test_percent == 0:
        percent_change = np.inf
    else:
        percent_change = max(test_percent, ref_percent) / min(test_percent, ref_percent)
    if percent_change < ratio_change_thres:
        return 1

    # if somehow the data is small or has a zero frequency in it, use fisher. Otherwise chi2
    if ref_count + test_count > 100 and (contingency_matrix_df.values != 0).all():
        _, p_val, *_ = chi2_contingency(contingency_matrix_df.values)
    else:
        _, p_val = fisher_exact(contingency_matrix_df.values)

    return p_val


def dominant_frequency_change(validation_dataset: Dataset, train_dataset: Dataset, p_val_thres: float = 0.0001, dominance_ratio: float = 2,  ratio_change_thres: float = 1.5):
    """Detect values highly represented in the tested and reference data and checks if their relative and absolute percentage have increased significantly

    Args:
        train_dataset (Dataset): The training dataset object. Must contain an index.
        validation_dataset (Dataset): The validation dataset object. Must contain an index.
        p_val_thres (float = 0.0001): Maximal p-value to pass the statistical test determining if the value abundance has changed significantly (0-1).
        dominance_ratio (float = 2): Next most abundance value has to be THIS times less than the first (0-inf).
        ratio_change_thres (float = 1.5): The dominant frequency has to change by at least this ratio (0-inf).
    Returns:
        CheckResult:  result value is dataframe that containes the dominant value change for each column.

    Raises:
        MLChecksValueError: If the object is not a Dataset or DataFrame instance

    """
    validation_dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
    validation_dataset.validate_shared_features(train_dataset, dominant_frequency_change.__name__)

    columns = train_dataset.features()

    train_f = train_dataset.data
    val_f = validation_dataset.data

    val_len = len(val_f)
    train_len = len(train_f)
    p_df = {}

    for column in columns:
        top_val = val_f[column].value_counts()
        top_train = train_f[column].value_counts()
        
        if(top_val.iloc[0] > top_val.iloc[1] * dominance_ratio):
            p_val = find_p_val(top_val.index[0], top_train, top_val, train_len, val_len, ratio_change_thres)
            if p_val < p_val_thres:
                p_df[column] = {'value': top_val.index[0], 'p value': p_val}
        elif(top_train.iloc[0] > top_train.iloc[1] * dominance_ratio):
            p_val = find_p_val(top_val.index[0], top_train, top_val, train_len, val_len, ratio_change_thres)
            if p_val < p_val_thres:
                p_df[column] = {'value': top_train.index[0], 'p value': p_val}

    p_df = pd.DataFrame.from_dict(p_df, orient='index')
    
    return CheckResult(p_df, header='Data Sample Leakage Report',
                       check=dominant_frequency_change, display=p_df)

class DominantFrequencyChange(TrainValidationBaseCheck):
    """Finds dominant frequency change."""

    def run(self, validation_dataset: Dataset, train_dataset: Dataset) -> CheckResult:
        """Run dominant_frequency_change_report check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
            p_val_thres (float = 0.0001): Maximal p-value to pass the statistical test determining if the value abundance has changed significantly (0-1).
            dominance_ratio (float = 2): Next most abundance value has to be THIS times less than the first (0-inf).
            ratio_change_thres (float = 1.5): The dominant frequency has to change by at least this ratio (0-inf).
        Returns:
            CheckResult: Detects values highly represented in the tested and reference data and checks if their
                         relative and absolute percentage have increased significantly and makes a report in a dataframe.
        """
        return dominant_frequency_change(validation_dataset=validation_dataset, train_dataset=train_dataset, **self.params)
