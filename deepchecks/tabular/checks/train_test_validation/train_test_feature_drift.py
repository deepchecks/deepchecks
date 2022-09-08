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
"""Module contains Train Test Drift check."""

from collections import OrderedDict
from typing import Dict, List, Union

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceFeatureMixin
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.utils.distribution.drift import calc_drift_and_plot, drift_condition, get_drift_plot_sidenote
from deepchecks.utils.typing import Hashable

__all__ = ['TrainTestFeatureDrift']


@docstrings
class TrainTestFeatureDrift(TrainTestCheck, ReduceFeatureMixin):
    """
    Calculate drift between train dataset and test dataset per feature, using statistical measures.

    Check calculates a drift score for each column in test dataset, by comparing its distribution to the train
    dataset.

    For numerical columns, we use the Earth Movers Distance.
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
    categorical_drift_method: str, default: "cramer_v"
        decides which method to use on categorical variables. Possible values are:
        "cramer_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    ignore_na: bool, default True
        For categorical columns only. If True, ignores nones for categorical drift. If False, considers none as a
        separate category. For numerical columns we always ignore nones.
    aggregation_method: str, default: 'l2_weighted'
        {feature_aggregation_method_argument:2*indent}
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
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='cramer_v',
            ignore_na: bool = True,
            aggregation_method='l2_weighted',
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
        self.categorical_drift_method = categorical_drift_method
        self.ignore_na = ignore_na
        self.aggregation_method = aggregation_method
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

        values_dict = OrderedDict()
        displays_dict = OrderedDict()

        features_order = (
            tuple(
                feature_importance
                .sort_values(ascending=False)
                .index
            )
            if feature_importance is not None
            else None
        )

        for column in train_dataset.features:
            if column in train_dataset.numerical_features:
                column_type = 'numerical'
            elif column in train_dataset.cat_features:
                column_type = 'categorical'
            else:
                continue  # we only support categorical or numerical features
            if feature_importance is not None:
                fi_rank = features_order.index(column) + 1
                plot_title = f'{column} (#{int(fi_rank)} in FI)'
            else:
                plot_title = column

            value, method, display = calc_drift_and_plot(
                train_column=train_dataset.data[column],
                test_column=test_dataset.data[column],
                value_name=column,
                column_type=column_type,
                plot_title=plot_title,
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
            values_dict[column] = {
                'Drift score': value,
                'Method': method,
                'Importance': feature_importance[column] if feature_importance is not None else None
            }
            displays_dict[column] = display

        if context.with_display:
            sorted_by = self.sort_feature_by
            if self.sort_feature_by == 'feature importance' and feature_importance is not None:
                features_order = [feat for feat in features_order if feat in values_dict]
                columns_order = features_order[:self.n_top_columns]
            elif self.sort_feature_by == 'drift + importance' and feature_importance is not None:
                feature_columns = [feat for feat in features_order if feat in values_dict]
                feature_columns.sort(key=lambda col: values_dict[col]['Drift score'] + values_dict[col]['Importance'],
                                     reverse=True)
                columns_order = feature_columns[:self.n_top_columns]
                sorted_by = 'the sum of the drift score and the feature importance'
            else:
                columns_order = sorted(values_dict.keys(), key=lambda col: values_dict[col]['Drift score'],
                                       reverse=True)[:self.n_top_columns]
                sorted_by = 'drift score'

            headnote = [f"""<span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the features, sorted
                by {sorted_by} and showing only the top {self.n_top_columns} features, according to {sorted_by}.
            </span>""", get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by),
                        'If available, the plot titles also show the feature importance (FI) rank']

            displays = headnote + [displays_dict[col] for col in columns_order
                                   if col in train_dataset.cat_features + train_dataset.numerical_features]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Train Test Feature Drift')

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        feature_importance = [column_info['Importance'] for column_info in check_result.value.values()]
        feature_importance = None if None in feature_importance else feature_importance
        values = pd.Series({column: info['Drift score'] for column, info in check_result.value.items()})
        return self.feature_reduce(self.aggregation_method, values, feature_importance, 'Drift Score')

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.2,
                                            max_allowed_numeric_score: float = 0.1,
                                            allowed_num_features_exceeding_threshold: int = 0):
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            The max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.1
            The max threshold for the numeric variable drift score
        allowed_num_features_exceeding_threshold: int , default: 0
            Determines the number of features with drift score above threshold needed to fail the condition.

        Returns
        -------
        ConditionResult
            False if more than allowed_num_features_exceeding_threshold drift scores are above threshold, True otherwise
        """
        condition = drift_condition(max_allowed_categorical_score, max_allowed_numeric_score, 'column', 'columns',
                                    allowed_num_features_exceeding_threshold)

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score}',
                                  condition)
