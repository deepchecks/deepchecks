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
"""The single_feature_contribution check module."""
import typing as t

import deepchecks.ppscore as pps
from deepchecks import CheckResult, Dataset, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.plot import create_colorbar_barchart_for_check
from deepchecks.utils.typing import Hashable


__all__ = ['SingleFeatureContributionTrainTest']

FC = t.TypeVar('FC', bound='SingleFeatureContributionTrainTest')


class SingleFeatureContributionTrainTest(TrainTestBaseCheck):
    """
    Return the difference in PPS (Predictive Power Score) of all features between train and test datasets.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    When we compare train PPS to test PPS, A high difference can strongly indicate leakage,
    as a feature that was "powerful" in train but not in test can be explained by leakage in train that does
     not affect a new dataset.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        ppscore_params (dict): dictionary of additional parameters for the ppscore predictor function
        n_show_top (int): Number of features to show, sorted by the magnitude of difference in PPS
    """

    def __init__(self, ppscore_params=None, n_show_top: int = 5):
        super().__init__()
        self.ppscore_params = ppscore_params
        self.n_show_top = n_show_top

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Returns:
            CheckResult:
                value is a dictionary with PPS difference per feature column.
                data is a bar graph of the PPS of each feature.

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._single_feature_contribution_train_test(train_dataset=train_dataset,
                                                            test_dataset=test_dataset)

    def _single_feature_contribution_train_test(self, train_dataset: Dataset, test_dataset: Dataset):
        train_dataset = Dataset.validate_dataset(train_dataset)
        train_dataset.validate_label()
        test_dataset = Dataset.validate_dataset(test_dataset)
        test_dataset.validate_label()
        features_names = train_dataset.validate_shared_features(test_dataset)
        label_name = train_dataset.validate_shared_label(test_dataset)
        ppscore_params = self.ppscore_params or {}

        relevant_columns = features_names + [label_name]
        df_pps_train = pps.predictors(df=train_dataset.data[relevant_columns], y=train_dataset.label_name,
                                      random_seed=42,
                                      **ppscore_params)
        df_pps_test = pps.predictors(df=test_dataset.data[relevant_columns],
                                           y=test_dataset.label_name,
                                           random_seed=42, **ppscore_params)
        s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
        s_pps_test = df_pps_test.set_index('x', drop=True)['ppscore']
        s_difference = s_pps_train - s_pps_test

        def plot():
            # For display shows only positive differences
            s_difference_to_display = s_difference.apply(lambda x: 0 if x < 0 else x)
            s_difference_to_display = s_difference_to_display.sort_values(ascending=False).head(self.n_show_top)
            # Create graph:
            create_colorbar_barchart_for_check(x=s_difference_to_display.index, y=s_difference_to_display.values,
                                               ylabel='PPS Difference')

        text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is actually due '
                'to data',
                'leakage - meaning that the feature holds information that is based on the label to begin with.',
                '',
                'When we compare train PPS to test PPS, A high difference can strongly indicate leakage, as a '
                'feature',
                'that was powerful in train but not in test can be explained by leakage in train that is not '
                'relevant to a new dataset.']

        return CheckResult(value=s_difference.to_dict(), display=[plot, *text],
                           header='Single Feature Contribution Train-Test')

    def add_condition_feature_pps_difference_not_greater_than(self: FC, threshold: float = 0.2) -> FC:
        """Add new condition.

        Add condition that will check that difference between train
        dataset feature pps and test dataset feature pps is not greater than X.

        Args:
            threshold: train test ps difference upper bound
        """
        def condition(value: t.Dict[Hashable, float]) -> ConditionResult:
            failed_features = [
                feature_name
                for feature_name, pps_diff in value.items()
                if pps_diff > threshold
            ]

            if failed_features:
                message = f'Features with PPS difference above threshold: {", ".join(map(str, failed_features))}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        pps_url = 'https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598'
        pps_html_url = f'<a href={pps_url}>Predictive Power Score</a>'

        return \
            self.add_condition(f'Train-Test features\' {pps_html_url} (PPS) difference is not greater than {threshold}',
                               condition)
