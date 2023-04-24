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
"""Module contains Feature Drift check."""
from typing import Dict, List, Optional, Union

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.utils.abstracts.feature_drift import FeatureDriftAbstract
from deepchecks.utils.typing import Hashable

__all__ = ['FeatureDrift']


@docstrings
class FeatureDrift(TrainTestCheck, FeatureDriftAbstract, ReduceFeatureMixin):
    """
    Calculate drift between train dataset and test dataset per feature, using statistical measures.

    Check calculates a drift score for each column in test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Kolmogorov-Smirnov statistic.
    See https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    We also support Earth Mover's Distance (EMD).
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    For categorical variables, it is recommended to use Cramer's V, unless your variable includes categories with a
    small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.


    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all
        columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on
        columns variable.
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    sort_feature_by : str , default: "drift + importance"
        Indicates how features will be sorted. Possible values:
        - "feature importance":  sort features by feature importance.
        - "drift score": sort features by drift score.
        - "drift + importance": sort features by the sum of the drift score and the feature importance.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Only for categorical features. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots.
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
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    aggregation_method: Optional[str], default: 'l3_weighted'
        {feature_aggregation_method_argument:2*indent}
    min_samples : int , default: 10
        Minimum number of samples required to calculate the drift score. If there are not enough samples for either
        train or test, the check will return None for that feature. If there are not enough samples for all features,
        the check will raise a ``NotEnoughSamplesError`` exception.
    n_samples : int , default: 100_000
        Number of samples to use for drift computation and plot.
    random_state : int , default: 42
        Random seed for sampling.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_top_columns: int = 5,
            sort_feature_by: str = 'drift + importance',
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: Optional[int] = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            numerical_drift_method: str = 'KS',
            categorical_drift_method: str = 'cramers_v',
            ignore_na: bool = True,
            aggregation_method: Optional[str] = 'l3_weighted',
            min_samples: int = 10,
            n_samples: int = 100_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.margin_quantile_filter = margin_quantile_filter
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        if sort_feature_by in {'feature importance', 'drift score', 'drift + importance'}:
            self.sort_feature_by = sort_feature_by
        else:
            raise DeepchecksValueError(
                '"sort_feature_by must be either "feature importance", "drift score" or "drift + importance"'
            )
        self.n_top_columns = n_top_columns
        self.numerical_drift_method = numerical_drift_method
        self.categorical_drift_method = categorical_drift_method
        self.ignore_na = ignore_na
        self.aggregation_method = aggregation_method
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """
        Calculate drift for all columns.

        Parameters
        ----------
        context : Context
            The run context

        Returns
        -------
        CheckResult
            value: dictionary of column name to drift score.
            display: distribution graph for each column, comparing the train and test distributions.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset or DataFrame instance.
        """
        train_dataset: Dataset = context.train
        test_dataset: Dataset = context.test
        feature_importance = context.feature_importance
        train_dataset.assert_features()
        test_dataset.assert_features()

        train_dataset = train_dataset.select(
            self.columns, self.ignore_columns
        ).sample(self.n_samples, random_state=self.random_state)
        test_dataset = test_dataset.select(
            self.columns, self.ignore_columns
        ).sample(self.n_samples, random_state=self.random_state)

        features_order = (
            # In order to have consistent order for features with same importance, first sorting by index, and then
            # using mergesort which preserves the order of equal elements.
            tuple(feature_importance.sort_index(key=lambda x: x.astype(str))
                  .sort_values(kind='mergesort', ascending=False).index)
            if feature_importance is not None
            else None
        )

        common_columns = {}

        for column in train_dataset.features:
            if column in train_dataset.numerical_features:
                common_columns[column] = 'numerical'
            elif column in train_dataset.cat_features:
                common_columns[column] = 'categorical'
            else:
                # we only support categorical or numerical features
                continue

        results, displays = self._calculate_feature_drift(
            drift_kind='tabular-features',
            train=train_dataset.data,
            test=test_dataset.data,
            train_dataframe_name=train_dataset.name,
            test_dataframe_name=test_dataset.name,
            common_columns=common_columns,
            feature_importance=feature_importance,
            features_order=features_order,
            with_display=context.with_display
        )
        return CheckResult(
            value=results,
            display=displays,
            header='Feature Drift'
        )

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        feature_importance = pd.Series({column: info['Importance'] for column, info in check_result.value.items()})
        values = pd.Series({column: info['Drift score'] for column, info in check_result.value.items()})
        return self.feature_reduce(self.aggregation_method, values, feature_importance, 'Drift Score')
