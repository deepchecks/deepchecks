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
from deepchecks.base.check_context import CheckRunContext
from deepchecks.utils.plot import create_colorbar_barchart_for_check
from deepchecks.utils.typing import Hashable
from deepchecks.utils.strings import format_number
from deepchecks import CheckResult, SingleDatasetBaseCheck, ConditionResult


__all__ = ['SingleFeatureContribution']


FC = t.TypeVar('FC', bound='SingleFeatureContribution')

pps_url = 'https://docs.deepchecks.com/en/stable/examples/checks/methodology/single_feature_contribution_train_test' \
          '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


class SingleFeatureContribution(SingleDatasetBaseCheck):
    """Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore.predictors function
    n_show_top : int , default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    """

    def __init__(self, ppscore_params=None, n_show_top: int = 5):
        super().__init__()
        self.ppscore_params = ppscore_params or {}
        self.n_show_top = n_show_top

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary with PPS per feature column.
            data is a bar graph of the PPS of each feature.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        label_name = context.label_name
        features = context.features

        relevant_columns = features + [label_name]
        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=label_name, random_seed=42,
                                **self.ppscore_params)
        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        def plot(n_show_top=self.n_show_top):
            top_to_show = s_ppscore.head(n_show_top)
            # Create graph:
            create_colorbar_barchart_for_check(x=top_to_show.index, y=top_to_show.values)

        text = [
            'The Predictive Power Score (PPS) is used to estimate the ability of a feature to predict the '
            f'label by itself. (Read more about {pps_html})'
            'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is'
            ' actually due to data leakage - meaning that the feature holds information that is based on the label '
            'to begin with.']

        # display only if not all scores are 0
        display = [plot, *text] if s_ppscore.sum() else None

        return CheckResult(value=s_ppscore.to_dict(), display=display, header='Single Feature Contribution')

    def add_condition_feature_pps_not_greater_than(self: FC, threshold: float = 0.8) -> FC:
        """
        Add condition that will check that pps of the specified feature(s) is not greater than X.

        Parameters
        ----------
        threshold : float , default: 0.8
            pps upper bound
        Returns
        -------
        FC
        """
        def condition(value: t.Dict[Hashable, float]) -> ConditionResult:
            failed_features = {
                feature_name: format_number(pps_value)
                for feature_name, pps_value in value.items()
                if pps_value > threshold
            }

            if failed_features:
                message = f'Features with PPS above threshold: {failed_features}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Features\' Predictive Power Score is not greater than {format_number(threshold)}',
                                  condition)
