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

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.checks import ReduceMixin
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.distribution.drift import (SUPPORTED_CATEGORICAL_METHODS, SUPPORTED_NUMERIC_METHODS,
                                                 calc_drift_and_plot)

__all__ = ['TrainTestPredictionDrift']

from deepchecks.utils.strings import format_number


class TrainTestPredictionDrift(TrainTestCheck, ReduceMixin):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its distribution to the train
    dataset.
    For classification tasks, by default the drift score will be computed on the predicted probability of the positive
    (1) class for binary classification tasks, and on the predicted class itself for multiclass tasks. This behavior can
    be controlled using the `drift_mode` parameter.

    For numerical columns, we use the Earth Movers Distance.
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
        If  set to 'auto', compute drift on the predicted class if the task is multiclass, and on
        the predicted probability of the positive class if binary. Set to 'proba' to force drift on the predicted
        probabilities, and 'prediction' to force drift on the predicted classes. If set to 'proba', on a multiclass
        task, drift would be calculated on each class independently.
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
    categorical_drift_method: str, default: "cramer_v"
        decides which method to use on categorical variables. Possible values are:
        "cramer_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    aggregation_method: str, default: "max"
        Argument for the reduce_output functionality, decides how to aggregate the drift scores of different classes
        (for classification tasks) into a single score, when drift is computed on the class probabilities. Possible
        values are:
        'max': Maximum of all the class drift scores.
        'weighted': Weighted mean based on the class sizes in the train data set.
        'mean': Mean of all drift scores.
        'none': No averaging. Return a dict with a drift score for each class.
    max_classes_to_display: int, default: 3
        Max number of classes to show in the display when drift is computed on the class probabilities for
        classification tasks.
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            drift_mode: str = 'auto',
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method: str = 'cramer_v',
            ignore_na: bool = True,
            aggregation_method: str = 'max',
            max_classes_to_display: int = 3,
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(drift_mode, str):
            raise DeepchecksValueError('drift_mode must be a string')
        self.drift_mode = drift_mode.lower()
        if self.drift_mode not in ('auto', 'proba', 'prediction'):
            raise DeepchecksValueError('drift_mode must be one of "auto", "proba", "prediction"')
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
        self.ignore_na = ignore_na
        self.max_classes_to_display = max_classes_to_display
        self.aggregation_method = aggregation_method
        if self.aggregation_method not in ('weighted', 'mean', 'none', 'max'):
            raise DeepchecksValueError('aggregation_method must be one of "weighted", "mean", "none", "max"')

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        if (self.drift_mode == 'proba') and (context.task_type == TaskType.REGRESSION):
            raise DeepchecksValueError('probability_drift="proba" is not supported for regression tasks')

        train_dataset = context.train
        test_dataset = context.test
        model = context.model

        drift_score_dict, drift_display_dict = {}, {}
        method, classes = None, train_dataset.classes

        # Flag for computing drift on the probabilities rather than the predicted labels
        proba_drift = ((context.task_type == TaskType.BINARY) and (self.drift_mode == 'auto')) or \
                      (self.drift_mode == 'proba')

        if proba_drift:
            train_prediction = np.array(model.predict_proba(train_dataset.features_columns))
            test_prediction = np.array(model.predict_proba(test_dataset.features_columns))
            if test_prediction.shape[1] == 2:
                train_prediction = train_prediction[:, [1]]
                test_prediction = test_prediction[:, [1]]
        else:
            train_prediction = np.array(model.predict(train_dataset.features_columns)).reshape((-1, 1))
            test_prediction = np.array(model.predict(test_dataset.features_columns)).reshape((-1, 1))

        samples_per_class = train_dataset.label_col.value_counts().to_dict()

        for class_idx in range(train_prediction.shape[1]):
            class_name = classes[class_idx]
            drift_score_dict[class_name], method, drift_display_dict[class_name] = calc_drift_and_plot(
                train_column=pd.Series(train_prediction[:, class_idx].flatten()),
                test_column=pd.Series(test_prediction[:, class_idx].flatten()),
                value_name='model predictions' if not proba_drift else
                           f'predicted probabilities for class {class_name}',
                column_type='categorical' if (context.task_type != TaskType.REGRESSION) and (not proba_drift)
                            else 'numerical',
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                categorical_drift_method=self.categorical_drift_method,
                ignore_na=self.ignore_na,
                with_display=context.with_display,
            )

        if context.with_display:
            headnote = f"""<span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the predicted
                {'class probabilities' if proba_drift else 'classes'}.
            </span>"""

            # sort classes by their drift score
            displays = [headnote] + [x for _, x in sorted(zip(drift_score_dict.values(), drift_display_dict.values()),
                                                          reverse=True)][:self.max_classes_to_display]
        else:
            displays = None

        # Return float if single value (happens by default) or the whole dict if computing on probabilities for
        # multi-class tasks.
        values_dict = {
            'Drift score': drift_score_dict if len(drift_score_dict) > 1 else list(drift_score_dict.values())[0],
            'Method': method, 'Samples per class': samples_per_class}

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return prediction drift score."""
        if isinstance(check_result.value['Drift score'], float):
            return {'Prediction Drift Score': check_result.value['Drift score']}

        drift_values = list(check_result.value['Drift score'].values())
        if self.aggregation_method == 'none':
            return {f'Drift Score class {k}': v for k, v in check_result.value['Drift score'].items()}
        elif self.aggregation_method == 'mean':
            return {'Mean Drift Score': np.mean(drift_values)}
        elif self.aggregation_method == 'max':
            return {'Max Drift Score': np.max(drift_values)}
        elif self.aggregation_method == 'weighted':
            class_weight = np.array([check_result.value['Samples per class'][class_name] for class_name in
                                     check_result.value['Drift score'].keys()])
            class_weight = class_weight / np.sum(class_weight)
            return {'Weighted Drift Score': np.sum(np.array(drift_values) * class_weight)}

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.15,
                                            max_allowed_numeric_score: float = 0.075,
                                            max_allowed_psi_score: float = None,
                                            max_allowed_earth_movers_score: float = None):
        """
        Add condition - require drift score to be less than a certain threshold.

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
        max_allowed_psi_score: float, default None
            Deprecated. Please use max_allowed_categorical_score instead
        max_allowed_earth_movers_score: float, default None
            Deprecated. Please use max_allowed_numeric_score instead
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """
        if max_allowed_psi_score is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_allowed_psi_score is deprecated. please use '
                f'max_allowed_categorical_score instead',
                DeprecationWarning
            )
            if max_allowed_categorical_score is not None:
                max_allowed_categorical_score = max_allowed_psi_score
        if max_allowed_earth_movers_score is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_allowed_earth_movers_score is deprecated. please use '
                f'max_allowed_numeric_score instead',
                DeprecationWarning
            )
            if max_allowed_numeric_score is not None:
                max_allowed_numeric_score = max_allowed_earth_movers_score

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
