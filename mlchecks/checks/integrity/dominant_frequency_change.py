"""module contains Dominant Frequency Change check."""
from typing import  Dict

from scipy.stats import chi2_contingency, fisher_exact
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, CompareDatasetsBaseCheck

__all__ = ['dominant_frequency_change', 'DominantFrequencyChange']


def find_p_val(key: str, baseline_hist: Dict, test_hist: Dict, baseline_count: int,
               test_count: int, ratio_change_thres: float) -> float:
    """Find p value for column frequency change between the reference dataset to the test dataset.

    Args:
        key (str): key of the dominant value.
        baseline_hist (Dict): The baseline dataset histogram.
        test_hist (Dict): The test dataset histogram.
        baseline_count (int): The reference dataset row count.
        test_count (int): The test dataset row count.
        ratio_change_thres (float): The dominant frequency has to change by at least this ratio (0-inf).
    Returns:
        float: p value for the key.

    Raises:
        MLChecksValueError: If the object is not a Dataset or DataFrame instance

    """
    contingency_matrix_df = pd.DataFrame(np.zeros((2, 2)), index=['dominant', 'others'], columns=['ref', 'test'])
    contingency_matrix_df.loc['dominant', 'ref'] = baseline_hist.get(key, 0)
    contingency_matrix_df.loc['dominant', 'test'] = test_hist.get(key, 0)
    contingency_matrix_df.loc['others', 'ref'] = baseline_count - baseline_hist.get(key, 0)
    contingency_matrix_df.loc['others', 'test'] = test_count - test_hist.get(key, 0)

    test_percent = contingency_matrix_df.loc['dominant', 'test'] / test_count
    baseline_percent = contingency_matrix_df.loc['dominant', 'ref'] / baseline_count
    if baseline_percent == 0 or test_percent == 0:
        percent_change = np.inf
    else:
        percent_change = max(test_percent, baseline_percent) / min(test_percent, baseline_percent)
    if percent_change < ratio_change_thres:
        return None

    # if somehow the data is small or has a zero frequency in it, use fisher. Otherwise chi2
    if baseline_count + test_count > 100 and (contingency_matrix_df.values != 0).all():
        _, p_val, *_ = chi2_contingency(contingency_matrix_df.values)
    else:
        _, p_val = fisher_exact(contingency_matrix_df.values)

    return p_val


def dominant_frequency_change(dataset: Dataset, baseline_dataset: Dataset,
                              p_value_threshold: float = 0.0001, dominance_ratio: float = 2,
                              ratio_change_thres: float = 1.5):
    """Check if dominant values have increased significantly between test and reference data.

    Args:
        dataset (Dataset): The dataset object. Must contain an index.
        baseline_dataset (Dataset): The baseline dataset object. Must contain an index.
        p_value_threshold (float = 0.0001): Maximal p-value to pass the statistical test determining
                                      if the value abundance has changed significantly (0-1).
        dominance_ratio (float = 2): Next most abundance value has to be THIS times less than the first (0-inf).
        ratio_change_thres (float = 1.5): The dominant frequency has to change by at least this ratio (0-inf).
    Returns:
        CheckResult:  result value is dataframe that containes the dominant value change for each column.

    Raises:
        MLChecksValueError: If the object is not a Dataset or DataFrame instance

    """
    baseline_dataset = Dataset.validate_dataset_or_dataframe(baseline_dataset)
    dataset = Dataset.validate_dataset_or_dataframe(dataset)
    dataset.validate_shared_features(baseline_dataset, dominant_frequency_change.__name__)

    columns = baseline_dataset.features()

    test_df = dataset.data
    baseline_df = baseline_dataset.data

    baseline_len = len(baseline_df)
    test_len = len(test_df)
    p_df = {}

    for column in columns:
        top_ref = baseline_df[column].value_counts()
        top_test = test_df[column].value_counts()

        if len(top_ref) ==1 or top_ref.iloc[0] > top_ref.iloc[1] * dominance_ratio:
            value = top_ref.index[0]
            p_val = find_p_val(value, top_test, top_ref, test_len, baseline_len, ratio_change_thres)
            if p_val and p_val < p_value_threshold:
                count_ref = top_ref[value]
                count_test = top_test.get(value, 0)
                p_df[column] = {'Value': value,
                                'Reference data': f'{count_ref} ({count_ref / baseline_len * 100:0.2f})',
                                'Tested data': f'{count_test} ({count_test / test_len * 100:0.2f})'}
        elif len(top_test) ==1 or top_test.iloc[0] > top_test.iloc[1] * dominance_ratio:
            value = top_test.index[0]
            p_val = find_p_val(value, top_test, top_ref, test_len, baseline_len, ratio_change_thres)
            if p_val and p_val < p_value_threshold:
                count_test = top_test[value]
                count_ref = top_ref.get(value, 0)
                p_df[column] = {'Value': value,
                                'Reference data': f'{count_ref} ({count_ref / baseline_len * 100:0.2f}%)',
                                'Tested data': f'{count_test} ({count_test / test_len * 100:0.2f}%)'}

    p_df = pd.DataFrame.from_dict(p_df, orient='index') if len(p_df) else None

    return CheckResult(p_df, header='Data Sample Leakage Report',
                       check=dominant_frequency_change, display=p_df)

class DominantFrequencyChange(CompareDatasetsBaseCheck):
    """Finds dominant frequency change."""

    def run(self, dataset, baseline_dataset) -> CheckResult:
        """Run dominant_frequency_change_report check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            validation_dataset (Dataset): The validation dataset object. Must contain an index.
            p_value_threshold (float = 0.0001): Maximal p-value to pass the statistical test
                                          determining if the value abundance has changed significantly (0-1).
            dominance_ratio (float = 2): Next most abundance value has to be THIS times less than the first (0-inf).
            ratio_change_thres (float = 1.5): The dominant frequency has to change by at least this ratio (0-inf).
        Returns:
            CheckResult: Detects values highly represented in the tested and reference data and checks if their..
                         relative and absolute percentage have increased significantly and makes a report.
        """
        return dominant_frequency_change(dataset=dataset, baseline_dataset=baseline_dataset, **self.params)
