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
"""The feature label correlation check module."""
import typing as t

import deepchecks.ppscore as pps
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.check_utils.feature_label_correlation_utils import get_pps_figure, pd_series_to_trace
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

__all__ = ['FeatureLabelCorrelation']


FLC = t.TypeVar('FLC', bound='FeatureLabelCorrelation')


pps_url = 'https://docs.deepchecks.com/en/stable/checks_gallery/tabular/' \
          'train_test_validation/plot_feature_label_correlation_change.html'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


class FeatureLabelCorrelation(SingleDatasetCheck):
    """Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore.predictors function
    n_top_features : int , default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    random_state : int , default: None
        Random state for the ppscore.predictors function
    """

    def __init__(
        self,
        ppscore_params=None,
        n_top_features: int = 5,
        random_state: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ppscore_params = ppscore_params or {}
        self.n_top_features = n_top_features
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
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
        dataset = context.get_data_by_kind(dataset_kind)
        dataset.assert_features()
        relevant_columns = dataset.features + [dataset.label_name]

        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name, random_seed=self.random_state,
                                **self.ppscore_params)
        s_ppscore = df_pps.set_index('x', drop=True)['ppscore']

        if context.with_display:
            top_to_show = s_ppscore.head(self.n_top_features)

            fig = get_pps_figure(per_class=False, n_of_features=len(top_to_show))
            fig.add_trace(pd_series_to_trace(top_to_show, dataset_kind.value))

            text = [
                'The Predictive Power Score (PPS) is used to estimate the ability of a feature to predict the '
                f'label by itself (Read more about {pps_html}).'
                ' A high PPS (close to 1) can mean that this feature\'s success in predicting the label is'
                ' actually due to data leakage - meaning that the feature holds information that is based on the label '
                'to begin with.']

            # display only if not all scores are 0
            display = [fig, *text] if s_ppscore.sum() else None
        else:
            display = None

        return CheckResult(value=s_ppscore.to_dict(), display=display, header='Feature Label Correlation')

    def add_condition_feature_pps_less_than(self: FLC, threshold: float = 0.8) -> FLC:
        """
        Add condition that will check that pps of the specified feature(s) is less than the threshold.

        Parameters
        ----------
        threshold : float , default: 0.8
            pps upper bound
        Returns
        -------
        FLC
        """
        def condition(value: t.Dict[Hashable, float]) -> ConditionResult:
            failed_features = {
                feature_name: format_number(pps_value)
                for feature_name, pps_value in value.items()
                if pps_value >= threshold
            }

            if failed_features:
                message = f'Found {len(failed_features)} out of {len(value)} features with PPS above threshold: ' \
                          f'{failed_features}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(value))

        return self.add_condition(f'Features\' Predictive Power Score is less than {format_number(threshold)}',
                                  condition)
