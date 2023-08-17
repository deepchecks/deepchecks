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
"""Module containing the prediction popularity drift check."""
import typing as t

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.recommender import Context
from deepchecks.tabular import SingleDatasetCheck
from deepchecks.utils.distribution.drift import calc_drift_and_plot

__all__ = ['PredictionPopularityDrift']


class PredictionPopularityDrift(SingleDatasetCheck):
    """Compute popularity drift between predictions and true labels, using statistical measures.

    Check calculates a drift score for the prediction in the test dataset, by comparing its
    distribution to the predictions.

    For numerical columns, we use the Kolmogorov-Smirnov statistic.
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

    **Note:** In case of highly imbalanced classes, it is recommended to use Cramer's V, together with setting
    the ``balance_classes`` parameter to ``True``. This also requires setting the ``drift_mode`` parameter to
    ``auto`` (default) or ``'prediction'``.


    Parameters
    ----------
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles)
        of the distribution will be filtered out of the EMD calculation.
        This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number
        are binned into an "Other" category. Ignored if balance_classes=True.
    max_num_categories_for_drift: int, default: None
        Only relevant if drift is calculated for classification predictions.
        Max number of allowed categories. If there are more, they are binned into
        an "Other" category.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs,
        as the number of shown categories is limited by max_num_categories_for_display.
        Possible values:
        - 'train_largest': Show the largest prediction categories.
        - 'test_largest': Show the largest labels categories.
        - 'largest_difference': Show the largest difference between categories.
    numerical_drift_method: str, default: "KS"
        decides which method to use on numerical variables. Possible values are:
        "EMD" for Earth Mover's Distance (EMD), "KS" for Kolmogorov-Smirnov (KS).
    balance_classes: bool, default: False
        If True, all categories will have an equal weight in the Cramer's V score.
        This is useful when the categorical variable is highly imbalanced, and we want
        to be alerted on changes in proportion to the category size, and not only to the
        entire dataset. Must have categorical_drift_method = "cramers_v" and
        drift_mode = "auto" or "prediction".
        If True, the variable frequency plot will be created with a log scale in the y-axis.
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift.
        If False, considers none as a separate category.
        For numerical columns we always ignore nones.
    aggregation_method: t.Optional[str], default: "max"
        Argument for the reduce_output functionality, decides how to aggregate the drift scores
        of different classes (for classification tasks) into a single score,
        when drift is computed on the class probabilities. Possible values are:
        'max': Maximum of all the class drift scores.
        'weighted': Weighted mean based on the class sizes in the prediction data set.
        'mean': Mean of all drift scores.
        None: No averaging. Return a dict with a drift score for each class.
    min_samples : int , default: 10
        Minimum number of samples required to calculate the drift score. If there are not enough
        samples for either predictions or labels,
        the check will raise a ``NotEnoughSamplesError`` exception.
    n_samples : int , default: 100_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
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
            ignore_na: bool = True,
            aggregation_method: t.Optional[str] = 'max',
            min_samples: t.Optional[int] = 10,
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
        self.aggregation_method = aggregation_method
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.random_state = random_state
        if self.aggregation_method not in ('weighted', 'mean', 'none', None, 'max'):
            raise DeepchecksValueError('aggregation_method must be one of "weighted", "mean", "max", None')

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Compute popularity drift.

        Returns
        -------
        CheckResult
            value: drift score.
            display: distribution graph, comparing the  popularity distribution of predictions and true labels.
        """
        test_labels = context.get_data_by_kind(dataset_kind).label_col
        sample_size = min(self.n_samples, len(test_labels))

        test_labels = test_labels.sample(sample_size, random_state=self.random_state)
        test_pred = context.model.predictions

        interaction_dataset = context.get_interaction_dataset
        item_id = interaction_dataset.item_index_name
        item_popularity = interaction_dataset.data[item_id].value_counts().to_dict()

        pred_popularity = [item_popularity[item] for sub in test_pred for item in sub if item in item_popularity]
        label_popularity = [item_popularity[item] for sub in test_labels for item in sub if item in item_popularity]

        drift_score, _, drift_display = calc_drift_and_plot(
            train_column=pd.Series(pred_popularity),
            test_column=pd.Series(label_popularity),
            value_name='Prediction Popularity',
            column_type='numerical',
            plot_title='Prediction Popularity Drift',
            margin_quantile_filter=self.margin_quantile_filter,
            max_num_categories_for_drift=self.max_num_categories_for_drift,
            min_category_size_ratio=self.min_category_size_ratio,
            max_num_categories_for_display=self.max_num_categories_for_display,
            show_categories_by=self.show_categories_by,
            numerical_drift_method=self.numerical_drift_method,
            categorical_drift_method=self.categorical_drift_method,
            balance_classes=self.balance_classes,
            ignore_na=self.ignore_na,
            min_samples=self.min_samples,
            raise_min_samples_error=True,
            dataset_names=('Prediction', 'True Labels'),
            with_display=context.with_display,
        )

        return CheckResult(
            value=drift_score,
            header='Prediction Popularity Drift',
            display=drift_display,
        )
