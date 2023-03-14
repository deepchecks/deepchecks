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
"""Module containing the label recommendation drift check."""
import typing as t

import pandas as pd

from deepchecks.core import CheckResult

from deepchecks.core.errors import DeepchecksValueError

from deepchecks.core.reduce_classes import ReduceMixin
from deepchecks.recommender import Context
from deepchecks.tabular import SingleDatasetCheck
from deepchecks.utils.distribution.drift import calc_drift_and_plot


class LabelRecommendationDrift(SingleDatasetCheck):

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
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        model = context.model

        labels = dataset.label_col
        predictions = model.predict(dataset.features_columns)
        # flatten sequence of sequences to a single sequence
        flattened_predictions = [item for sublist in predictions for item in sublist]

        drift_score, method, drift_display = calc_drift_and_plot(
            train_column=pd.Series(labels),
            test_column=pd.Series(flattened_predictions),
            value_name='Items',
            column_type='categorical',
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
            dataset_names=('Selected Items', 'Recommended Items'),
            with_display=context.with_display,
        )

        return CheckResult(
            value=drift_score,
            header=f'Label Recommendation Drift',
            display=drift_display,
        )