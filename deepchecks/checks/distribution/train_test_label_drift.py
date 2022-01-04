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
"""Module contains Train Test label Drift check."""

from typing import Dict
from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult

__all__ = ['TrainTestLabelDrift']

from deepchecks.checks.distribution.dist_utils import calc_drift_and_plot


class TrainTestLabelDrift(TrainTestBaseCheck):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset.
    For numerical columns, we use the Earth Movers Distance.
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf
    For categorical columns, we use the Population Stability Index (PSI).
    See https://en.wikipedia.org/wiki/Wasserstein_metric.


    Args:
        max_num_categories (int):
            Only for categorical columns. Max number of allowed categories. If there are more,
            they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
            for both drift calculation and for distribution plots.
    """

    def __init__(
            self,
            max_num_categories: int = 10
    ):
        super().__init__()
        self.max_num_categories = max_num_categories

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: not used in this check.

        Returns:
            CheckResult:
                value: dictionary of column name to drift score.
                display: distribution graph for each column, comparing the train and test distributions.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        return self._calc_drift(train_dataset, test_dataset)

    def _calc_drift(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
    ) -> CheckResult:
        """
        Calculate drift for all columns.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.

        Returns:
            CheckResult:
                value: drift score.
                display: label distribution graph, comparing the train and test distributions.
        """
        train_dataset = Dataset.validate_dataset(train_dataset)
        test_dataset = Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()

        drift_score, method, display = calc_drift_and_plot(
            train_column=train_dataset.label_col,
            test_column=test_dataset.label_col,
            plot_title=train_dataset.label_name,
            column_type='categorical' if train_dataset.label_type == 'classification_label' else 'numerical',
            max_num_categories=self.max_num_categories
        )

        headnote = """<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the label.
        </span>"""

        displays = [headnote, display]
        values_dict = {'Drift score': drift_score, 'Method': method}

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_psi_score: float = 0.2,
                                                   max_allowed_earth_movers_score: float = 0.1):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Earth movers does not have a common industry standard.

        Args:
            max_allowed_psi_score: the max threshold for the PSI score
            max_allowed_earth_movers_score: the max threshold for the Earth Mover's Distance score

        Returns:
            ConditionResult: False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            drift_score = result['Drift score']
            method = result['Method']
            has_failed = (drift_score > max_allowed_psi_score and method == 'PSI') or \
                         (drift_score > max_allowed_earth_movers_score and method == "Earth Mover's Distance")

            if method == 'PSI' and has_failed:
                return_str = f'Label has PSI over {max_allowed_psi_score} - Drift score is {drift_score:.2f}'
                return ConditionResult(False, return_str)
            elif method == "Earth Mover's Distance" and has_failed:
                return_str = f'Label has Earth Mover\'s Distance over {max_allowed_earth_movers_score} - ' \
                             f'Drift score is {drift_score:.2f}'
                return ConditionResult(False, return_str)

            return ConditionResult(True)

        return self.add_condition(f'PSI and Earth Mover\'s Distance for label drift cannot be greater than '
                                  f'{max_allowed_psi_score} or {max_allowed_earth_movers_score} respectively',
                                  condition)
