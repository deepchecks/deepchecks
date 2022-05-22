# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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
from copy import copy

import numpy as np

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.check_utils.single_feature_contribution_utils import \
    get_single_feature_contribution
from deepchecks.core.condition import ConditionCategory
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

__all__ = ['SingleFeatureContributionTrainTest']

FC = t.TypeVar('FC', bound='SingleFeatureContributionTrainTest')

pps_url = 'https://docs.deepchecks.com/en/stable/examples/tabular/' \
          'checks/methodology/single_feature_contribution_train_test' \
          '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


class SingleFeatureContributionTrainTest(TrainTestCheck):
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

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore predictor function
    n_top_features : int , default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    random_state : int , default: None
        Random state for the ppscore.predictors function
    min_pps_to_show: float, default 0.05
        Minimum PPS to show a class in the graph
    """

    def __init__(self, ppscore_params=None,
                 n_top_features: int = 5,
                 random_state: int = None,
                 min_pps_to_show: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.ppscore_params = ppscore_params or {}
        self.n_top_features = n_top_features
        self.random_state = random_state
        self.min_pps_to_show = min_pps_to_show

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary with PPS difference per feature column.
            data is a bar graph of the PPS of each feature.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        train_dataset = context.train
        test_dataset = context.test

        train_dataset.assert_features()
        train_dataset.assert_label()
        relevant_columns = train_dataset.features + [train_dataset.label_name]

        text = [
            'The Predictive Power Score (PPS) is used to estimate the ability of a feature to predict the '
            f'label by itself. (Read more about {pps_html})'
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
            '   An anomalous value, could indicate drift in test dataset that caused a coincidental correlation to '
            'the target label.'
        ]

        ret_value, display = get_single_feature_contribution(train_dataset.data[relevant_columns],
                                                             train_dataset.label_name,
                                                             test_dataset.data[relevant_columns],
                                                             test_dataset.label_name, self.ppscore_params,
                                                             self.n_top_features,
                                                             min_pps_to_show=self.min_pps_to_show,
                                                             random_state=self.random_state)

        if display:
            display += text

        return CheckResult(value=ret_value, display=display, header='Single Feature Contribution Train-Test')

    def add_condition_feature_pps_difference_not_greater_than(self: FC, threshold: float = 0.2,
                                                              include_negative_diff: bool = True) -> FC:
        """Add new condition.

        Add condition that will check that difference between train
        dataset feature pps and test dataset feature pps is not greater than X.

        Parameters
        ----------
        threshold: float, default: 0.2
            train test pps difference upper bound.
        include_negative_diff: bool, default True
            This parameter decides whether the condition checks the absolute value of the difference, or just the
            positive value.
            The difference is calculated as train PPS minus test PPS. This is because we're interested in the case
            where the test dataset is less predictive of the label than the train dataset, as this could indicate
            leakage of labels into the train dataset.
        """

        def condition(value: t.Dict[Hashable, t.Dict[Hashable, float]]) -> ConditionResult:

            diff_dict = copy(value['train-test difference'])
            if include_negative_diff is True:
                diff_dict = {k: np.abs(v) for k, v in diff_dict.items()}

            failed_features = {
                feature_name: format_number(pps_diff)
                for feature_name, pps_diff in diff_dict.items()
                if pps_diff > threshold
            }

            if failed_features:
                message = f'Found {len(failed_features)} features with PPS difference above threshold out of ' \
                          f'{len(value)} features: {failed_features}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS, f'Passed for {len(diff_dict)} relevant features')

        return self.add_condition(f'Train-Test features\' Predictive Power Score difference is not greater than '
                                  f'{format_number(threshold)}', condition)

    def add_condition_feature_pps_in_train_not_greater_than(self: FC, threshold: float = 0.7) -> FC:
        """Add new condition.

        Add condition that will check that train dataset feature pps is not greater than X.

        Parameters
        ----------
        threshold : float , default: 0.7
            pps upper bound

        Returns
        -------
        FC
        """

        def condition(value: t.Dict[Hashable, t.Dict[Hashable, float]]) -> ConditionResult:
            failed_features = {
                feature_name: format_number(pps_value)
                for feature_name, pps_value in value['train'].items()
                if pps_value > threshold
            }

            if failed_features:
                message = f'Found {len(failed_features)} features in train dataset with PPS above threshold out' \
                          f' of {len(value["train"])} features: {failed_features}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS, f'Passed for {len(value["train"])} relevant features')

        return self.add_condition(f'Train features\' Predictive Power Score is not greater than '
                                  f'{format_number(threshold)}', condition)
