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
"""Module contains the Abstract cass for Prediction Drift checks."""
import numpy as np
import pandas as pd

from deepchecks import CheckResult
from deepchecks.utils.distribution.drift import calc_drift_and_plot, get_drift_plot_sidenote


class PredictionDriftAbstract:
    """Abstract class for prediction drift checks."""

    def prediction_drift(self, train_prediction, test_prediction, model_classes, with_display,
                         proba_drift, cat_plot) -> CheckResult:
        """Calculate prediction drift.

        Args:
            train_prediction : np.ndarray
                train prediction or probabilities
            test_prediction : np.ndarray
                test prediction or probabilities
            model_classes : list
                list of model classes
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
