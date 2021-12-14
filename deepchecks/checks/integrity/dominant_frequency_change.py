# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""module contains Dominant Frequency Change check."""
from typing import Dict, Optional

from scipy.stats import chi2_contingency, fisher_exact
import numpy as np
import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df
from deepchecks.utils.strings import format_percent
from deepchecks.errors import DeepchecksValueError


__all__ = ['DominantFrequencyChange']


class DominantFrequencyChange(TrainTestBaseCheck):
    """Check if dominant values have increased significantly between test and reference data.

    Args:
        dominance_ratio (float = 2):
            Next most abundant value has to be THIS times less than the first (0-inf).
        ratio_change_thres (float = 1.5):
            The dominant frequency has to change by at least this ratio (0-inf).
        n_top_columns (int):
            (optional - used only if model was specified)
            amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(self, dominance_ratio: float = 2, ratio_change_thres: float = 1.5,
                 n_top_columns: int = 10):
        super().__init__()
        self.dominance_ratio = dominance_ratio
        self.ratio_change_thres = ratio_change_thres
        self.n_top_columns = n_top_columns

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            test_dataset (Dataset): The test dataset object. Must contain an index.

        Returns:
            CheckResult: Detects values highly represented in the tested and reference data and checks if their..
            relative and absolute percentage have increased significantly and makes a report.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        feature_importances = calculate_feature_importance_or_null(test_dataset, model)
        return self._dominant_frequency_change(train_dataset=train_dataset, test_dataset=test_dataset,
                                               feature_importances=feature_importances)

    def _find_p_val(self, key: str, baseline_hist: Dict, test_hist: Dict, baseline_count: int,
                    test_count: int, ratio_change_thres: float) -> Optional[float]:
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
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance

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
            return

        # if somehow the data is small or has a zero frequency in it, use fisher. Otherwise chi2
        if baseline_count + test_count > 100 and (contingency_matrix_df.values != 0).all():
            _, p_val, *_ = chi2_contingency(contingency_matrix_df.values)
        else:
            _, p_val = fisher_exact(contingency_matrix_df.values)

        return p_val

    def _dominant_frequency_change(self, train_dataset: Dataset, test_dataset: Dataset,
                                   feature_importances: pd.Series = None):
        """Run the check logic.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain an index.
            test_dataset (Dataset): The test dataset object. Must contain an index.
        Returns:
            CheckResult: result value is dict that contains the dominant value change for each column.
        """
        test_dataset = Dataset.validate_dataset_or_dataframe(test_dataset)
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        test_dataset.validate_shared_features(train_dataset)

        columns = train_dataset.features

        test_df = test_dataset.data
        baseline_df = train_dataset.data

        baseline_len = len(baseline_df)
        test_len = len(test_df)
        p_dict = {}

        for column in columns:
            top_ref = baseline_df[column].value_counts(dropna=False)
            top_test = test_df[column].value_counts(dropna=False)

            if len(top_ref) == 1 or top_ref.iloc[0] > top_ref.iloc[1] * self.dominance_ratio:
                value = top_ref.index[0]
                p_val = self._find_p_val(value, top_test, top_ref, test_len, baseline_len, self.ratio_change_thres)
                if p_val:
                    count_ref = top_ref[value]
                    count_test = top_test.get(value, 0)
                    p_dict[column] = {'Value': value,
                                      'Reference data %': count_ref / baseline_len * 100,
                                      'Tested data %': count_test / test_len * 100,
                                      'Reference data #': count_ref,
                                      'Tested data #': count_test,
                                      'P value': p_val}
            elif len(top_test) == 1 or top_test.iloc[0] > top_test.iloc[1] * self.dominance_ratio:
                value = top_test.index[0]
                p_val = self._find_p_val(value, top_test, top_ref, test_len, baseline_len, self.ratio_change_thres)
                if p_val:
                    count_test = top_test[value]
                    count_ref = top_ref.get(value, 0)
                    p_dict[column] = {'Value': value,
                                      'Reference data %': count_ref / baseline_len * 100,
                                      'Tested data %': count_test / test_len * 100,
                                      'Reference data #': count_ref,
                                      'Tested data #': count_test,
                                      'P value': p_val}

        if len(p_dict):
            sorted_p_df = pd.DataFrame.from_dict(p_dict, orient='index')
            sorted_p_df = column_importance_sorter_df(
                sorted_p_df,
                test_dataset,
                feature_importances,
                self.n_top_columns
            )
        else:
            sorted_p_df = None

        return CheckResult(p_dict, display=sorted_p_df)

    def add_condition_p_value_not_less_than(self, p_value_threshold: float = 0.0001):
        """Add condition - require min p value allowed per column.

        Args:
            p_value_threshold (float = 0.0001): Minimal p-value to pass the statistical test determining
              if the value abundance has changed significantly (0-1).

        """

        def condition(result: Dict) -> ConditionResult:
            failed_columns = []
            for column, values in result.items():
                p_val = values['P value']
                if p_val < p_value_threshold:
                    failed_columns.append(column)
            if failed_columns:
                return ConditionResult(False,
                                       f'Found columns with low p-value: {failed_columns}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'P value not less than {p_value_threshold}',
                                  condition)

    def add_condition_ratio_of_change_not_more_than(self, percent_change_threshold: float = 0.25):
        """Add condition - require change in the ratio of the dominant value to be below the threshold.

        Args:
            percent_change_threshold (float): The maximal change in the ratio out of data between training data and
             test data that the dominant value is allowed to change
        """
        if percent_change_threshold < 0 or percent_change_threshold > 1:
            raise DeepchecksValueError(f'percent_change_threshold should be between 0 and 1,'
                                       f' found {percent_change_threshold}')

        def condition(result: Dict) -> ConditionResult:
            failed_columns = []
            for column, values in result.items():
                diff = values['Tested data %'] - values['Reference data %']
                if diff > percent_change_threshold:
                    failed_columns.append(column)
            if failed_columns:
                return ConditionResult(False,
                                       f'Found columns with high change in dominant value: {failed_columns}')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Change in ratio of dominant value in data not more'
                                  f' than {format_percent(percent_change_threshold)}',
                                  condition)
