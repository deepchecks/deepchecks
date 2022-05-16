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

import warnings
from typing import Dict

import pandas as pd

from deepchecks import ConditionCategory
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.distribution.drift import (SUPPORTED_CATEGORICAL_METHODS,
                                                 SUPPORTED_NUMERICAL_METHODS,
                                                 calc_drift_and_plot)

__all__ = ['TrainTestPredictionDrift']


class TrainTestPredictionDrift(TrainTestCheck):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for categorical columns. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If None, there is no limit.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    categorical_drift_method: str, default: "Cramer"
        Cramer for Cramer's V, PSI for Population Stability Index (PSI).
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='Cramer',
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        self.margin_quantile_filter = margin_quantile_filter
        if max_num_categories is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_num_categories is deprecated. please use max_num_categories_for_drift '
                'and max_num_categories_for_display instead',
                DeprecationWarning
            )
            max_num_categories_for_drift = max_num_categories_for_drift or max_num_categories
            max_num_categories_for_display = max_num_categories_for_display or max_num_categories
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.categorical_drift_method = categorical_drift_method

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
            margin_quantile_filter=self.margin_quantile_filter,
            max_num_categories_for_drift=self.max_num_categories_for_drift,
            max_num_categories_for_display=self.max_num_categories_for_display,
            show_categories_by=self.show_categories_by,
            categorical_drift_method=self.categorical_drift_method,
        )

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the predictions.
        </span>"""

        displays = [headnote, display]
        values_dict = {'Drift score': drift_score, 'Method': method}

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_categorical_score: float = 0.15,
                                                   max_allowed_numeric_score: float = 0.075):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.
        The threshold was lowered by 25% compared to feature drift defaults due to the higher importance of prediction
        drift.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            the max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.1
            the max threshold for the numeric variable drift score
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            drift_score = result['Drift score']
            method = result['Method']
            has_failed = (drift_score > max_allowed_categorical_score and method in SUPPORTED_CATEGORICAL_METHODS) or \
                         (drift_score > max_allowed_numeric_score and method in SUPPORTED_NUMERICAL_METHODS)

            if has_failed:
                return_str = f'Found model prediction {method} above threshold: {drift_score:.2f}'
                return ConditionResult(ConditionCategory.FAIL, return_str)

            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'categorical drift score <= {max_allowed_categorical_score} and '
                                  f'numerical drift score <= {max_allowed_numeric_score}',
                                  condition)
