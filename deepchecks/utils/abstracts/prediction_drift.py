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
"""The base abstract functionality for prediction drift checks."""
import abc
import typing as t

import numpy as np
import pandas as pd

from deepchecks import CheckResult, ConditionCategory, ConditionResult
from deepchecks.utils.distribution.drift import calc_drift_and_plot, get_drift_plot_sidenote
from deepchecks.utils.strings import format_number

__all__ = ['PredictionDriftAbstract']


class PredictionDriftAbstract(abc.ABC):
    """Base class for prediction drift checks."""

    drift_mode: str = 'auto'
    margin_quantile_filter: float = 0.025
    max_num_categories_for_drift: int = None
    min_category_size_ratio: float = 0.01
    max_num_categories_for_display: int = 10
    show_categories_by: str = 'largest_difference'
    numerical_drift_method: str = 'KS'
    categorical_drift_method: str = 'cramers_v'
    balance_classes: bool = False
    ignore_na: bool = True
    aggregation_method: t.Optional[str] = 'max'
    max_classes_to_display: int = 3
    min_samples: t.Optional[int] = 10
    n_samples: int = 100_000
    random_state: int = 42
    add_condition: t.Callable[..., t.Any]

    def _prediction_drift(self, train_prediction, test_prediction, model_classes, with_display,
                          proba_drift, cat_plot) -> CheckResult:
        """Calculate prediction drift.

        Args:
            train_prediction : np.ndarray
                train prediction or probabilities
            test_prediction : np.ndarray
                test prediction or probabilities
            model_classes : List[str]
                List of model classes names
            with_display : bool
                flag for displaying the prediction distribution graph
            proba_drift : bool
                flag for computing drift on the probabilities rather than the predicted labels
            cat_plot : bool
                flag for plotting the distribution of the predictions as a categorical plot

        CheckResult
            value: drift score.
            display: prediction distribution graph, comparing the train and test distributions.
        """
        drift_score_dict, drift_display_dict = {}, {}
        method = None

        if proba_drift:
            if test_prediction.shape[1] == 2:
                train_prediction = train_prediction[:, [1]]
                test_prediction = test_prediction[:, [1]]

            # Get the classes in the same order as the model's predictions
            train_converted_from_proba = train_prediction.argmax(axis=1)
            test_converted_from_proba = test_prediction.argmax(axis=1)
            samples_per_class = pd.Series(np.concatenate([train_converted_from_proba, test_converted_from_proba], axis=0
                                                         ).squeeze()).value_counts().sort_index()

            # If label exists, get classes from it and map the samples_per_class index to these classes
            if model_classes is not None:
                classes = model_classes
                class_dict = dict(zip(range(len(classes)), classes))
                samples_per_class.index = samples_per_class.index.to_series().map(class_dict).values
            else:
                classes = list(sorted(samples_per_class.keys()))
            samples_per_class = samples_per_class.to_dict()
        else:
            # Get the classes in the same order as the model's predictions
            samples_per_class = pd.Series(np.concatenate([train_prediction, test_prediction], axis=0
                                                         ).squeeze()).value_counts().to_dict()
            classes = list(sorted(samples_per_class.keys()))

        has_min_samples = hasattr(self, 'min_samples')
        additional_kwargs = {}
        if has_min_samples:
            additional_kwargs['min_samples'] = self.min_samples

        for class_idx in range(train_prediction.shape[1]):
            class_name = classes[class_idx]
            drift_score_dict[class_name], method, drift_display_dict[class_name] = calc_drift_and_plot(
                train_column=pd.Series(train_prediction[:, class_idx].flatten()),
                test_column=pd.Series(test_prediction[:, class_idx].flatten()),
                value_name='model predictions' if not proba_drift else
                f'predicted probabilities for class {class_name}',
                column_type='categorical' if cat_plot else 'numerical',
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                min_category_size_ratio=self.min_category_size_ratio,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                numerical_drift_method=self.numerical_drift_method,
                categorical_drift_method=self.categorical_drift_method,
                balance_classes=self.balance_classes,
                ignore_na=self.ignore_na,
                raise_min_samples_error=has_min_samples,
                with_display=with_display,
                **additional_kwargs
            )

        if with_display:
            headnote = [f"""<span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the predicted
                {'class probabilities' if proba_drift else 'classes'}.
            </span>""", get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by)]

            # sort classes by their drift score
            displays = headnote + [x for _, x in sorted(zip(drift_score_dict.values(), drift_display_dict.values()),
                                                        reverse=True)][:self.max_classes_to_display]
        else:
            displays = None

        # Return float if single value (happens by default) or the whole dict if computing on probabilities for
        # multi-class tasks.
        values_dict = {
            'Drift score': drift_score_dict if len(drift_score_dict) > 1 else list(drift_score_dict.values())[0],
            'Method': method, 'Samples per class': samples_per_class}

        return CheckResult(value=values_dict, display=displays, header='Prediction Drift')

    def add_condition_drift_score_less_than(self, max_allowed_drift_score: float = 0.15):
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.

        Parameters
        ----------
        max_allowed_drift_score: float , default: 0.15
            the max threshold for the categorical variable drift score
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: t.Dict) -> ConditionResult:
            drift_score_dict = result['Drift score']
            # Move to dict for easier looping
            if not isinstance(drift_score_dict, dict):
                drift_score_dict = {0: drift_score_dict}
            method = result['Method']
            has_failed = {}
            drift_score = 0
            for class_name, drift_score in drift_score_dict.items():
                has_failed[class_name] = drift_score > max_allowed_drift_score

            if len(has_failed) == 1:
                details = f'Found model prediction {method} drift score of {format_number(drift_score)}'
            else:
                details = f'Found {sum(has_failed.values())} classes with model predicted probability {method} drift' \
                          f' score above threshold: {max_allowed_drift_score}.'

            category = ConditionCategory.FAIL if any(has_failed.values()) else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(f'Prediction drift score < {max_allowed_drift_score}', condition)
