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
"""Module contains Train Test label Drift check."""

from typing import Dict

import pandas as pd

from deepchecks import ConditionCategory
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.utils.distribution.drift import calc_drift_and_plot


__all__ = ['TrainTestPredictionDrift']


class TrainTestPredictionDrift(TrainTestCheck):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its distribution to the train
    dataset.
    For numerical columns, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric
    For categorical columns, we use the Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    max_num_categories : int , default: 10
        Only for categorical columns. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
        for both drift calculation and for distribution plots.
    """

    def __init__(
        self,
        max_num_categories: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_num_categories = max_num_categories

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        train_dataset = context.train
        test_dataset = context.test
        model = context.model

        train_prediction = model.predict(train_dataset.features_columns)
        test_prediction = model.predict(test_dataset.features_columns)

        drift_score, method, display = calc_drift_and_plot(
            train_column=pd.Series(train_prediction),
            test_column=pd.Series(test_prediction),
            value_name='model predictions',
            column_type='categorical' if train_dataset.label_type == 'classification_label' else 'numerical',
            max_num_categories=self.max_num_categories
        )

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the predictions.
        </span>"""

        displays = [headnote, display]
        values_dict = {'Drift score': drift_score, 'Method': method}

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_psi_score: float = 0.15,
                                                   max_allowed_earth_movers_score: float = 0.075):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Earth movers does not have a common industry standard.
        The threshold was lowered by 25% compared to feature drift defaults due to the higher importance of prediction
        drift.

        Parameters
        ----------
        max_allowed_psi_score: float , default: 0.2
            the max threshold for the PSI score
        max_allowed_earth_movers_score: float ,  default: 0.1
            the max threshold for the Earth Mover's Distance score
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            drift_score = result['Drift score']
            method = result['Method']
            has_failed = (drift_score > max_allowed_psi_score and method == 'PSI') or \
                         (drift_score > max_allowed_earth_movers_score and method == "Earth Mover's Distance")

            if method == 'PSI' and has_failed:
                return_str = f'Found model prediction PSI above threshold: {drift_score:.2f}'
                return ConditionResult(ConditionCategory.FAIL, return_str)
            elif method == "Earth Mover's Distance" and has_failed:
                return_str = f'Model Prediction\'s Earth Mover\'s Distance above threshold: {drift_score:.2f}'
                return ConditionResult(ConditionCategory.FAIL, return_str)

            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'PSI <= {max_allowed_psi_score} and Earth Mover\'s Distance <= '
                                  f'{max_allowed_earth_movers_score} for model prediction drift',
                                  condition)
