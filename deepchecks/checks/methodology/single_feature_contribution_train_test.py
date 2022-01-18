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
import numpy as np
import plotly.graph_objects as go

import deepchecks.ppscore as pps
from deepchecks import CheckResult, Dataset, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.plot import colors
from deepchecks.utils.typing import Hashable
from deepchecks.utils.strings import format_number

__all__ = ['SingleFeatureContributionTrainTest']

FC = t.TypeVar('FC', bound='SingleFeatureContributionTrainTest')

pps_url = 'https://docs.deepchecks.com/en/stable/examples/checks/methodology/single_feature_contribution_train_test' \
          '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'
pps_html_url = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


class SingleFeatureContributionTrainTest(TrainTestBaseCheck):
    """
    Return the Predictive Power Score of all features, in order to estimate each feature's ability to predict the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability of each feature to predict the label.
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
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset)
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset)
        label_name = self._datasets_share_label([train_dataset, test_dataset])
        features_names = self._datasets_share_features([train_dataset, test_dataset])

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

        s_difference_to_display = np.abs(s_difference).apply(lambda x: 0 if x < 0 else x)
        s_difference_to_display = s_difference_to_display.sort_values(ascending=False).head(self.n_show_top)

        s_pps_train_to_display = s_pps_train[s_difference_to_display.index]
        s_pps_test_to_display = s_pps_test[s_difference_to_display.index]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=s_pps_train_to_display.index,
                             y=s_pps_train_to_display,
                             name='Train',
                             marker_color=colors['Train'], text=s_pps_train_to_display.round(2), textposition='outside'
                             ))
        fig.add_trace(go.Bar(x=s_pps_test_to_display.index,
                             y=s_pps_test_to_display,
                             name='Test',
                             marker_color=colors['Test'], text=s_pps_test_to_display.round(2), textposition='outside'
                             ))
        fig.add_trace(go.Scatter(x=s_difference_to_display.index,
                                 y=s_difference_to_display,
                                 name='Train-Test Difference (abs)',
                                 marker=dict(symbol='circle', size=15),
                                 line=dict(color='#aa57b5', width=5)
                                 ))

        fig.update_layout(
            title='Predictive Power Score (PPS) - Can a feature predict the label by itself?',
            xaxis_title='Column',
            yaxis_title='Predictive Power Score (PPS)',
            yaxis_range=[0, 1.05],
            legend=dict(x=1.0, y=1.0),
            barmode='group',
            width=800, height=500
        )

        text = [
            f'The PPS ({pps_html_url}) is used to estimate the ability of a feature to predict the label by itself.'
            '',
            '<u>In the graph above</u>, we should suspect we have problems in our data if:',
            ''
            '1. <b>Train dataset PPS values are high</b>:',
            'Can indicate that this feature\'s success in predicting the label is actually due to data leakage, ',
            '   meaning that the feature holds information that is based on the label to begin with.',
            '2. <b>Large difference between train and test PPS</b> (train PPS is larger):',
            '   An even more powerful indication of data leakage, as a feature that was powerful in train but not in '
            'test ',
            '   can be explained by leakage in train that is not relevant to a new dataset.',
            '3. <b>Large difference between test and train PPS</b> (test PPS is larger):',
            '   An anomalous value, could indicate  drift in test dataset that caused a coincidental correlation to '
            'the target label.'
        ]

        ret_value = {'train': s_pps_train.to_dict(), 'test': s_pps_test.to_dict(),
                     'train-test difference': s_difference.to_dict()}

        # display only if not all scores are 0
        display = [fig, *text] if s_pps_train.sum() or s_pps_test.sum() else None

        return CheckResult(value=ret_value, display=display, header='Single Feature Contribution Train-Test')

    def add_condition_feature_pps_difference_not_greater_than(self: FC, threshold: float = 0.2) -> FC:
        """Add new condition.

        Add condition that will check that difference between train
        dataset feature pps and test dataset feature pps is not greater than X.

        Args:
            threshold: train test ps difference upper bound
        """

        def condition(value: t.Dict[Hashable, t.Dict[Hashable, float]]) -> ConditionResult:
            failed_features = {
                feature_name: format_number(pps_diff)
                for feature_name, pps_diff in value['train-test difference'].items()
                if pps_diff > threshold
            }

            if failed_features:
                message = f'Features with PPS difference above threshold: {failed_features}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test features\' {pps_html_url} (PPS) difference is not greater than '
                                  f'{format_number(threshold)}', condition)

    def add_condition_feature_pps_in_train_not_greater_than(self: FC, threshold: float = 0.7) -> FC:
        """Add new condition.

        Add condition that will check that train dataset feature pps is not greater than X.

        Args:
            threshold: pps upper bound
        """

        def condition(value: t.Dict[Hashable, t.Dict[Hashable, float]]) -> ConditionResult:
            failed_features = {
                feature_name: format_number(pps_value)
                for feature_name, pps_value in value['train'].items()
                if pps_value > threshold
            }

            if failed_features:
                message = f'Features in train dataset with PPS above threshold: {failed_features}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return \
            self.add_condition(f'Train features\' {pps_html_url} (PPS) is not greater than {format_number(threshold)}',
                               condition)
