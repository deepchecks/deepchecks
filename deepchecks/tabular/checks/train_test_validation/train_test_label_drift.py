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

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.reduce_classes import ReduceLabelMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.distribution.drift import (SUPPORTED_CATEGORICAL_METHODS, SUPPORTED_NUMERIC_METHODS,
                                                 calc_drift_and_plot, get_drift_plot_sidenote)

__all__ = ['TrainTestLabelDrift']

from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.strings import format_number


class TrainTestLabelDrift(TrainTestCheck, ReduceLabelMixin):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical labels, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.


    Parameters
    ----------
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Only for classification. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots
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
    ignore_na: bool, default False
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    n_samples : int , default: 100_000
        Number of samples to use for drift computation and plot.
    random_state : int , default: 42
        Random seed for sampling.
    """

    def __init__(
            self,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='cramer_v',
            ignore_na: bool = False,
            n_samples: int = 100_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.categorical_drift_method = categorical_drift_method
        self.ignore_na = ignore_na
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)

        drift_score, method, display = calc_drift_and_plot(
            train_column=train_dataset.label_col,
            test_column=test_dataset.label_col,
            value_name=train_dataset.label_name,
            column_type='categorical' if context.task_type != TaskType.REGRESSION else 'numerical',
            margin_quantile_filter=self.margin_quantile_filter,
            max_num_categories_for_drift=self.max_num_categories_for_drift,
            min_category_size_ratio=self.min_category_size_ratio,
            max_num_categories_for_display=self.max_num_categories_for_display,
            show_categories_by=self.show_categories_by,
            categorical_drift_method=self.categorical_drift_method,
            ignore_na=self.ignore_na,
            with_display=context.with_display,
            dataset_names=(train_dataset.name, test_dataset.name)
        )

        values_dict = {'Drift score': drift_score, 'Method': method}

        if context.with_display:
            displays = ["""<span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the label.
            </span>""", get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by), display]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return label drift score."""
        return {'Label Drift Score': check_result.value['Drift score']}

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        return False

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.2,
                                            max_allowed_numeric_score: float = 0.1):
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.

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
                         (drift_score > max_allowed_numeric_score and method in SUPPORTED_NUMERIC_METHODS)

            details = f'Label\'s drift score {method} is {format_number(drift_score)}'
            category = ConditionCategory.FAIL if has_failed else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score} for label drift',
                                  condition)
