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
"""The calibration_metric check module."""
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.metrics import ModelType, task_type_validation

__all__ = ["CalibrationMetric"]


class CalibrationMetric(SingleDatasetBaseCheck):
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
        return self._calibration_metric(dataset, model)

    def _calibration_metric(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        task_type_validation(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        ds_x = dataset.features_columns
        ds_y = dataset.label_col
        y_pred = model.predict_proba(ds_x)

        briers_scores = {}
        unique_labels = dataset.label_col.unique()

        if len(unique_labels) == 2:
            briers_scores[0] = brier_score_loss(ds_y, y_pred[:, 1])
        else:
            for n_class in unique_labels:
                prob_pos = y_pred[:, n_class]
                clf_score = brier_score_loss(ds_y == n_class, prob_pos, pos_label=n_class)
                briers_scores[n_class] = clf_score

        def display():
            fig = plt.figure(figsize=(6, 6))
            ax1 = fig.add_subplot(111)

            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            # If it's a binary classification
            if len(unique_labels) == 2:
                fraction_of_positives, mean_predicted_value = calibration_curve(ds_y, y_pred[:, 1], n_bins=10)

                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                         label=f"(brier={briers_scores[0]:9.4f})")
            else:
                for n_class in unique_labels:
                    prob_pos = y_pred[:, n_class]

                    fraction_of_positives, mean_predicted_value = \
                        calibration_curve(ds_y == n_class, prob_pos, n_bins=10)

                    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                             label=f"{n_class} (brier={briers_scores[n_class]:9.4f})")

                ax1.set_ylabel("Fraction of positives")
                ax1.set_ylim([-0.05, 1.05])
                ax1.set_title("Calibration plots  (reliability curve)")
                ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), )

                ax1.set_xlabel("Mean predicted value")

            plt.tight_layout()

        calibration_text = "Calibration curves (also known as reliability diagrams) compare how well the " \
                           "probabilistic predictions of a binary classifier are calibrated. It plots the true " \
                           "frequency of the positive label against its predicted probability, for binned predictions."
        brier_text = "The Brier score metric may be used to assess how well a classifier is calibrated. For more " \
                     "info, please visit https://en.wikipedia.org/wiki/Brier_score"
        return CheckResult(briers_scores, header="Calibration Metric",
                           display=[calibration_text, display, brier_text])
