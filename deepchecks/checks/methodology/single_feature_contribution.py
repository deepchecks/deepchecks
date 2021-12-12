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
from deepchecks.utils.plot import create_colorbar_barchart_for_check
from deepchecks.utils.typing import Hashable
from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult


__all__ = ['SingleFeatureContribution']


FC = t.TypeVar('FC', bound='SingleFeatureContribution')


class SingleFeatureContribution(SingleDatasetBaseCheck):
    """Return the PPS (Predictive Power Score) of all features in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        ppscore_params (dict): dictionary of additional parameters for the ppscore.predictors function
    """

    def __init__(self, ppscore_params=None, n_show_top: int = 5):
        super().__init__()
        self.ppscore_params = ppscore_params
        self.n_show_top = n_show_top

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            CheckResult:
                value is a dictionary with PPS per feature column.
                data is a bar graph of the PPS of each feature.

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._single_feature_contribution(dataset=dataset)

    def _single_feature_contribution(self, dataset: Dataset):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        ppscore_params = self.ppscore_params or {}

        relevant_columns = dataset.features + [dataset.label_name]
        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name, random_seed=42,
                                **ppscore_params)
        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        def plot(n_show_top=self.n_show_top):
            top_to_show = s_ppscore.head(n_show_top)
            # Create graph:
            create_colorbar_barchart_for_check(x=top_to_show.index, y=top_to_show.values)

        text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                'A high PPS (close to 1) can mean that this feature\'s success in predicting the label is'
                ' actually due to data',
                'leakage - meaning that the feature holds information that is based on the label to begin with.']

        return CheckResult(value=s_ppscore.to_dict(), display=[plot, *text], header='Single Feature Contribution')

    def add_condition_feature_pps_not_greater_than(self: FC, threshold: float = 0.8) -> FC:
        """
        Add condition that will check that pps of the specified feature(s) is not greater than X.

        Args:
            threshold: pps upper bound
        """
        def condition(value: t.Dict[Hashable, float]) -> ConditionResult:
            failed_features = [
                feature_name
                for feature_name, pps_value in value.items()
                if pps_value > threshold
            ]

            if failed_features:
                message = f'Features with PPS above threshold: {", ".join(map(str, failed_features))}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        pps_url = 'https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598'
        pps_html_url = f'<a href={pps_url}>Predictive Power Score</a>'

        return self.add_condition(f'Features\' {pps_html_url} (PPS) is not greater than {threshold}', condition)
