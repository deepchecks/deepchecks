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

import warnings
from collections import OrderedDict
from typing import Dict, List, Union

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.distribution.drift import (SUPPORTED_CATEGORICAL_METHODS, SUPPORTED_NUMERIC_METHODS,
                                                 calc_drift_and_plot, get_drift_method)
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

__all__ = ['TrainTestFeatureDrift']


class TrainTestFeatureDrift(TrainTestCheck):
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
    sort_feature_by : str , default: feature importance
        Indicates how features will be sorted. Can be either "feature importance"
        or "drift score"
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for categorical columns. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If None, there is no limit.
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
        "cramers_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    n_samples : int , default: 100_000
        Number of samples to use for drift computation and plot.
    random_state : int , default: 42
        Random seed for sampling.
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_top_columns: int = 5,
            sort_feature_by: str = 'feature importance',
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='cramer_v',
            n_samples: int = 100_000,
            random_state: int = 42,
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.margin_quantile_filter = margin_quantile_filter
        if max_num_categories is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_num_categories is deprecated. please use max_num_categories_for_drift '
                'and max_num_categories_for_display instead',
                DeprecationWarning
            )
            max_num_categories_for_drift = max_num_categories_for_drift or max_num_categories
            max_num_categories_for_display = max_num_categories_for_display or max_num_categories
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        if sort_feature_by in {'feature importance', 'drift score'}:
            self.sort_feature_by = sort_feature_by
        else:
            raise DeepchecksValueError('sort_feature_by must be either "feature importance" or "drift score"')
        self.n_top_columns = n_top_columns
        self.categorical_drift_method = categorical_drift_method
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
        features_importance = context.features_importance
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
                features_importance
                .sort_values(ascending=False)
                .index
            )
            if features_importance is not None
            else None
        )

        for column in train_dataset.features:
            if column in train_dataset.numerical_features:
                column_type = 'numerical'
            elif column in train_dataset.cat_features:
                column_type = 'categorical'
            else:
                continue  # we only support categorical or numerical features
            if features_importance is not None:
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
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                categorical_drift_method=self.categorical_drift_method,
            )
            values_dict[column] = {
                'Drift score': value,
                'Method': method,
                'Importance': features_importance[column] if features_importance is not None else None
            }
            displays_dict[column] = display

        if self.sort_feature_by == 'feature importance' and features_importance is not None:
            columns_order = features_order[:self.n_top_columns]
        else:
            columns_order = sorted(values_dict.keys(), key=lambda col: values_dict[col]['Drift score'],
                                   reverse=True)[:self.n_top_columns]

        sorted_by = self.sort_feature_by if features_importance is not None else 'drift score'

        headnote = f"""<span>
            The Drift score is a measure for the difference between two distributions, in this check - the test
            and train distributions.<br> The check shows the drift score and distributions for the features, sorted by
            {sorted_by} and showing only the top {self.n_top_columns} features, according to {sorted_by}.
            <br>If available, the plot titles also show the feature importance (FI) rank.
        </span>"""

        displays = [headnote] + [displays_dict[col] for col in columns_order
                                 if col in train_dataset.cat_features + train_dataset.numerical_features]

        return CheckResult(value=values_dict, display=displays, header='Train Test Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_categorical_score: float = 0.2,
                                                   max_allowed_numeric_score: float = 0.1,
                                                   number_of_top_features_to_consider: int = 5):
        """
        Add condition - require drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            the max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.1
            the max threshold for the numeric variable drift score
        number_of_top_features_to_consider: int , default: 5
            the number of top features for which exceed the threshold will fail the
            condition.
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            cat_method, num_method = get_drift_method(result)
            if all(x['Importance'] is not None for x in result.values()):
                columns_to_consider = \
                    [col_name for col_name, fi in sorted(result.items(), key=lambda item: item[1]['Importance'],
                                                         reverse=True)]
            else:
                columns_to_consider = \
                    [col_name for col_name, fi in sorted(result.items(), key=lambda item: item[1]['Drift score'],
                                                         reverse=True)]
            columns_to_consider = columns_to_consider[:number_of_top_features_to_consider]
            cat_drift_columns = {column: d['Drift score'] for column, d in result.items()
                                 if d['Method'] in SUPPORTED_CATEGORICAL_METHODS}
            cat_drift_columns = dict(sorted(cat_drift_columns.items(), key=lambda item: item[1], reverse=True))
            not_passing_categorical_columns = {column: format_number(d) for column, d in cat_drift_columns.items()
                                               if d > max_allowed_categorical_score and column in columns_to_consider}

            num_drift_columns = {column: d['Drift score'] for column, d in result.items()
                                 if d['Method'] in SUPPORTED_NUMERIC_METHODS}
            num_drift_columns = dict(sorted(num_drift_columns.items(), key=lambda item: item[1], reverse=True))
            not_passing_numeric_columns = {column: format_number(d) for column, d in num_drift_columns.items()
                                           if d > max_allowed_numeric_score and column in columns_to_consider}

            if not_passing_numeric_columns or not_passing_categorical_columns:
                details = f'Failed for {len(not_passing_categorical_columns) + len(not_passing_numeric_columns)} ' \
                          f'columns out of {len(result)}.'
                if not_passing_categorical_columns:
                    details += f' Found {len(not_passing_categorical_columns)} categorical columns with {cat_method} ' \
                               f'above threshold: {not_passing_categorical_columns}'
                if not_passing_numeric_columns:
                    details += f' Found {len(not_passing_numeric_columns)} numeric columns with {num_method} ' \
                               f'above threshold: {not_passing_numeric_columns}'
                return ConditionResult(ConditionCategory.FAIL, details)
            else:
                details = get_condition_passed_message(result) + '.'
                if cat_drift_columns:
                    column = next(iter(cat_drift_columns))
                    details += f' Column {column} has the highest categorical drift score: ' \
                               f'{format_number(cat_drift_columns[column])}'
                if num_drift_columns:
                    column = next(iter(num_drift_columns))
                    details += f' Column {column} has the highest numerical drift score: ' \
                               f'{format_number(num_drift_columns[column])}'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(f'categorical drift score <= {max_allowed_categorical_score} and '
                                  f'numerical drift score <= {max_allowed_numeric_score}',
                                  condition)
