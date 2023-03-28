# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains Prediction Drift check."""

from typing import Dict

import numpy as np

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp import Context, TrainTestCheck
from deepchecks.utils.abstracts.prediction_drift import PredictionDriftAbstract
from deepchecks.utils.distribution.drift import SUPPORTED_CATEGORICAL_METHODS, SUPPORTED_NUMERIC_METHODS
from deepchecks.utils.strings import format_number

__all__ = ['PredictionDrift']


class PredictionDrift(PredictionDriftAbstract, TrainTestCheck):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its distribution to the train
    dataset.
    For classification tasks, by default the drift score will be computed on the predicted probability of the positive
    (1) class for binary classification tasks, and on the predicted class itself for multiclass tasks. This behavior can
    be controlled using the `drift_mode` parameter.

    For numerical distributions, we use the Kolmogorov-Smirnov statistic.
    See https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    We also support Earth Mover's Distance (EMD).
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical predictions, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.


    Parameters
    ----------
    drift_mode: str, default: 'auto'
        For classification task, controls whether to compute drift on the predicted probabilities or the predicted
        classes. For regression task this parameter may be ignored.
        If set to 'auto', compute drift on the predicted class if the task is multiclass, and on
        the predicted probability of the positive class if binary. Set to 'proba' to force drift on the predicted
        probabilities, and 'prediction' to force drift on the predicted classes. If set to 'proba', on a multiclass
        task, drift would be calculated on each class independently.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Only relevant if drift is calculated for classification predictions. Max number of allowed categories.
        If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical predictions graph, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    numerical_drift_method: str, default: "KS"
        decides which method to use on numerical variables. Possible values are:
        "EMD" for Earth Mover's Distance (EMD), "KS" for Kolmogorov-Smirnov (KS).
    categorical_drift_method: str, default: "cramers_v"
        decides which method to use on categorical variables. Possible values are:
        "cramers_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    balance_classes: bool, default: False
        If True, all categories will have an equal weight in the Cramer's V score. This is useful when the categorical
        variable is highly imbalanced, and we want to be alerted on changes in proportion to the category size,
        and not only to the entire dataset. Must have categorical_drift_method = "cramers_v" and
        drift_mode = "auto" or "prediction".
        If True, the variable frequency plot will be created with a log scale in the y-axis.
    ignore_na: bool, default True
        For categorical predictions only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical predictions we always ignore nones.
    max_classes_to_display: int, default: 3
        Max number of classes to show in the display when drift is computed on the class probabilities for
        classification tasks.
    n_samples : int , default: 100_000
        number of samples to use for this check.
    """

    def __init__(
            self,
            drift_mode: str = 'auto',
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            numerical_drift_method: str = 'KS',
            categorical_drift_method: str = 'cramers_v',
            balance_classes: bool = False,
            ignore_na: bool = True,
            max_classes_to_display: int = 3,
            n_samples: int = 100_000,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(drift_mode, str):
            raise DeepchecksValueError('drift_mode must be a string')
        self.drift_mode = drift_mode.lower()
        if self.drift_mode not in ('auto', 'proba', 'prediction'):
            raise DeepchecksValueError('drift_mode must be one of "auto", "proba", "prediction"')
        if show_categories_by not in ('train_largest', 'test_largest', 'largest_difference'):
            raise DeepchecksValueError('show_categories_by must be one of "train_largest", "test_largest", '
                                       '"largest_difference"')
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.balance_classes = balance_classes
        self.ignore_na = ignore_na
        self.max_classes_to_display = max_classes_to_display
        self.n_samples = n_samples

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for predictions.

        Returns
        -------
        CheckResult
            value: drift score.
            display: prediction distribution graph, comparing the train and test distributions.
        """
        train_dataset = context.train.sample(self.n_samples, random_state=context.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=context.random_state)
        model = context.model

        # Flag for computing drift on the probabilities rather than the predicted labels
        proba_drift = ((len(context.model_classes) == 2) and (self.drift_mode == 'auto')) or \
                      (self.drift_mode == 'proba')

        if proba_drift:
            train_prediction = np.array(model.predict_proba(train_dataset))
            test_prediction = np.array(model.predict_proba(test_dataset))
        else:
            train_prediction = np.array(model.predict(train_dataset)).reshape((-1, 1))
            test_prediction = np.array(model.predict(test_dataset)).reshape((-1, 1))

        return self.prediction_drift(train_prediction, test_prediction, context.model_classes, context.with_display,
                                     proba_drift, not proba_drift)

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.15,
                                            max_allowed_numeric_score: float = 0.15):
        """
        Add condition - require drift score to be less than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.
        The threshold was lowered by 25% compared to property drift defaults due to the higher importance of prediction
        drift.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.15
            the max threshold for the categorical variable drift score
        max_allowed_numeric_score: float , default: 0.15
            the max threshold for the numeric variable drift score
        Returns
        -------
        ConditionResult
            False if any distribution has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            drift_score_dict = result['Drift score']
            # Move to dict for easier looping
            if not isinstance(drift_score_dict, dict):
                drift_score_dict = {0: drift_score_dict}
            method = result['Method']
            has_failed = {}
            drift_score = 0
            for class_name, drift_score in drift_score_dict.items():
                has_failed[class_name] = \
                    (drift_score >= max_allowed_categorical_score and method in SUPPORTED_CATEGORICAL_METHODS) or \
                    (drift_score >= max_allowed_numeric_score and method in SUPPORTED_NUMERIC_METHODS)

            if len(has_failed) == 1:
                details = f'Found model prediction {method} drift score of {format_number(drift_score)}'
            else:
                details = f'Found {sum(has_failed.values())} classes with model predicted probability {method} drift' \
                          f' score above threshold: {max_allowed_numeric_score}.'
            category = ConditionCategory.FAIL if any(has_failed.values()) else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score}',
                                  condition)
