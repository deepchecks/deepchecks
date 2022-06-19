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
"""The calibration score check module."""
import typing as t

import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from deepchecks.core import CheckResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.typing import ClassificationModel

__all__ = ['CalibrationScore']


class CalibrationScore(SingleDatasetCheck):
    """Calculate the calibration curve with brier score for each class."""

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is dictionary of a class and its brier score, displays the calibration curve
            graph with each class

        Raises
        ------
            DeepchecksValueError: If the data is not a Dataset instance with a label.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        context.assert_classification_task()
        ds_x = dataset.features_columns
        ds_y = dataset.label_col
        dataset_classes = dataset.classes
        model = t.cast(ClassificationModel, context.model)

        # Expect predict_proba to return in order of the sorted classes.
        y_pred = model.predict_proba(ds_x)

        briers_scores = {}

        if len(dataset_classes) == 2:
            briers_scores[0] = brier_score_loss(ds_y == dataset_classes[1], y_pred[:, 1])
        else:
            for class_index, class_name in enumerate(dataset_classes):
                prob_pos = y_pred[:, class_index]
                clf_score = brier_score_loss(ds_y == class_name, prob_pos)
                briers_scores[class_name] = clf_score

        if context.with_display:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                line_width=2, line_dash='dash',
                name='Perfectly calibrated',
            ))

            if len(dataset_classes) == 2:
                # Calibration curve must get labels of {0, 1} therefore in order to support other labels, apply mapping
                ds_y = ds_y.apply(lambda x: 0 if x == dataset_classes[0] else 1)
                fraction_of_positives, mean_predicted_value = calibration_curve(ds_y, y_pred[:, 1], n_bins=10)

                fig.add_trace(go.Scatter(
                    x=mean_predicted_value,
                    y=fraction_of_positives,
                    mode='lines+markers',
                    name=f'(brier:{briers_scores[0]:9.4f})',
                ))
            else:
                for class_index, class_name in enumerate(dataset_classes):
                    prob_pos = y_pred[:, class_index]

                    fraction_of_positives, mean_predicted_value = \
                        calibration_curve(ds_y == class_name, prob_pos, n_bins=10)

                    fig.add_trace(go.Scatter(
                        x=mean_predicted_value,
                        y=fraction_of_positives,
                        mode='lines+markers',
                        name=f'{class_name} (brier:{briers_scores[class_name]:9.4f})',
                    ))

            fig.update_layout(
                title_text='Calibration plots (reliability curve)',
                height=500
            )
            fig.update_yaxes(title='Fraction of positives')
            fig.update_xaxes(title='Mean predicted value')

            calibration_text = 'Calibration curves (also known as reliability diagrams) compare how well the ' \
                'probabilistic predictions of a binary classifier are calibrated. It plots the true ' \
                'frequency of the positive label against its predicted probability, for binned predictions.'
            brier_text = 'The Brier score metric may be used to assess how well a classifier is calibrated. For more ' \
                'info, please visit https://en.wikipedia.org/wiki/Brier_score'
            display = [calibration_text, fig, brier_text]
        else:
            display = None

        return CheckResult(briers_scores, header='Calibration Metric',
                           display=display)
