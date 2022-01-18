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
"""The calibration score check module."""
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import plotly.graph_objects as go

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.metrics import ModelType

__all__ = ['CalibrationScore']


class CalibrationScore(SingleDatasetBaseCheck):
    """Calculate the calibration curve with brier score for each class."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            dataset: a Dataset object
        Returns:
            CheckResult: value is dictionary of class and it's brier score, displays the calibration curve
            graph with each class

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._calibration_score(dataset, model)

    def _calibration_score(self, dataset: Dataset, model):
        dataset = Dataset.ensure_not_empty_dataset(dataset)
        ds_x = self._dataset_has_features(dataset)
        ds_y = self._dataset_has_label(dataset)

        self._verify_model_type(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        # Expect predict_proba to return in order of the sorted classes.
        y_pred = model.predict_proba(ds_x)

        briers_scores = {}

        if len(dataset.classes) == 2:
            briers_scores[0] = brier_score_loss(ds_y == dataset.classes[1], y_pred[:, 1])
        else:
            for class_index, class_name in enumerate(dataset.classes):
                prob_pos = y_pred[:, class_index]
                clf_score = brier_score_loss(ds_y == class_name, prob_pos)
                briers_scores[class_name] = clf_score

        fig = go.Figure()

        fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line_width=2, line_dash='dash',
                    name='Perfectly calibrated',
                ))

        if len(dataset.classes) == 2:
            fraction_of_positives, mean_predicted_value = calibration_curve(ds_y, y_pred[:, 1], n_bins=10)

            fig.add_trace(go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode='lines+markers',
                name=f'(brier:{briers_scores[0]:9.4f})',
            ))
        else:
            for class_index, class_name in enumerate(dataset.classes):
                prob_pos = y_pred[:, class_index]

                fraction_of_positives, mean_predicted_value = \
                    calibration_curve(ds_y == class_name, prob_pos, n_bins=10)

                fig.add_trace(go.Scatter(
                    x=mean_predicted_value,
                    y=fraction_of_positives,
                    mode='lines+markers',
                    name=f'{class_name} (brier:{briers_scores[class_name]:9.4f})',
                ))

        fig.update_layout(title_text='Calibration plots (reliability curve)',
                          width=700, height=500)
        fig.update_yaxes(title='Fraction of positives')
        fig.update_xaxes(title='Mean predicted value')

        calibration_text = 'Calibration curves (also known as reliability diagrams) compare how well the ' \
                           'probabilistic predictions of a binary classifier are calibrated. It plots the true ' \
                           'frequency of the positive label against its predicted probability, for binned predictions.'
        brier_text = 'The Brier score metric may be used to assess how well a classifier is calibrated. For more ' \
                     'info, please visit https://en.wikipedia.org/wiki/Brier_score'
        return CheckResult(briers_scores, header='Calibration Metric',
                           display=[calibration_text, fig, brier_text])
