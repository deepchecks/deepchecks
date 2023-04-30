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

"""Module contains Label Drift check."""

from typing import Dict

from deepchecks.core import CheckResult
from deepchecks.core.reduce_classes import ReduceLabelMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.abstracts.label_drift import LabelDriftAbstract

__all__ = ['LabelDrift']


class LabelDrift(TrainTestCheck, LabelDriftAbstract, ReduceLabelMixin):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Kolmogorov-Smirnov statistic.
    See https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    We also support Earth Mover's Distance (EMD).
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical labels, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.

    **Note:** In case of highly imbalanced classes, it is recommended to use Cramer's V, together with setting
    the ``balance_classes`` parameter to ``True``.

    Parameters
    ----------
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category. Ignored if balance_classes=True.
    max_num_categories_for_drift: int, default: None
        Only for classification. Max number of allowed categories. If there are more,
        they are binned into an "Other" category.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
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
        and not only to the entire dataset. Must have categorical_drift_method = "cramers_v".
        If True, the variable frequency plot will be created with a log scale in the y-axis.
    ignore_na: bool, default False
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    min_samples : int , default: 10
        Minimum number of samples required to calculate the drift score. If there are not enough samples for either
        train or test, the check will raise a ``NotEnoughSamplesError`` exception.
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
            numerical_drift_method: str = 'KS',
            categorical_drift_method: str = 'cramers_v',
            balance_classes: bool = False,
            ignore_na: bool = False,
            min_samples: int = 10,
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
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.balance_classes = balance_classes
        self.ignore_na = ignore_na
        self.min_samples = min_samples
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

        column_type = 'categorical' if context.task_type != TaskType.REGRESSION else 'numerical'

        return self._calculate_label_drift(train_dataset.label_col, test_dataset.label_col, train_dataset.label_name,
                                           column_type, context.with_display, (train_dataset.name, test_dataset.name))

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return label drift score."""
        return {'Label Drift Score': check_result.value['Drift score']}

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        return False
