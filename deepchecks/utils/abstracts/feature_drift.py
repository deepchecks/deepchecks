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
"""The base abstract functionality for features drift checks."""
import abc
import textwrap
import typing as t

import pandas as pd
from typing_extensions import Literal, Self

from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.utils.distribution.drift import calc_drift_and_plot, drift_condition, get_drift_plot_sidenote

__all__ = ['FeatureDriftAbstract']


class FeatureDriftAbstract(abc.ABC):
    """Base class for feature drift checks."""

    n_top_columns: int
    sort_feature_by: str
    margin_quantile_filter: float
    max_num_categories_for_drift: t.Optional[int]
    min_category_size_ratio: float
    max_num_categories_for_display: int
    show_categories_by: str
    numerical_drift_method: str
    categorical_drift_method: str
    ignore_na: bool
    min_samples: int
    n_samples: int
    add_condition: t.Callable[..., t.Any]

    def _calculate_feature_drift(
        self,
        drift_kind: Literal['tabular-features', 'nlp-properties'],
        train: pd.DataFrame,
        test: pd.DataFrame,
        common_columns: t.Dict[str, str],
        train_dataframe_name: str,
        test_dataframe_name: str,
        with_display: bool,
        override_plot_titles: t.Dict[str, str] = None,  # Specialized plot titles, as of now only for NLP
        feature_importance: t.Optional[pd.Series] = None,
        features_order: t.Optional[t.Sequence[str]] = None,
    ):
        plots = {}
        results = {}
        not_enough_samples = []

        for column_name, column_kind in common_columns.items():
            if features_order is not None:
                fi_rank = features_order.index(column_name) + 1
                plot_title = f'{column_name} (#{int(fi_rank)} in FI)'
            elif override_plot_titles is not None and column_name in override_plot_titles:
                plot_title = override_plot_titles[column_name]
            else:
                plot_title = column_name

            value, method, display = calc_drift_and_plot(
                train_column=train[column_name],
                test_column=test[column_name],
                value_name=column_name,
                column_type=column_kind,
                plot_title=plot_title,
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                min_category_size_ratio=self.min_category_size_ratio,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                numerical_drift_method=self.numerical_drift_method,
                categorical_drift_method=self.categorical_drift_method,
                ignore_na=self.ignore_na,
                min_samples=self.min_samples,
                with_display=with_display,
                dataset_names=(train_dataframe_name, test_dataframe_name)
            )

            if value == 'not_enough_samples':
                not_enough_samples.append(column_name)
                value = None
            else:
                plots[column_name] = display

            results[column_name] = {
                'Drift score': value,
                'Method': method,
                'Importance': (
                    feature_importance[column_name]
                    if feature_importance is not None
                    else None
                )
            }

        if len(not_enough_samples) == len(results.keys()):
            raise NotEnoughSamplesError(
                f'Not enough samples to calculate drift score. Minimum {self.min_samples} samples required. '
                'Note that for numerical columns, None values do not count as samples.'
                'Use the \'min_samples\' parameter to change this requirement.'
            )

        if not with_display:
            return results, []

        if self.sort_feature_by == 'feature importance' and features_order is not None:
            sorted_by = self.sort_feature_by
            features_order = [feat for feat in features_order if feat in results]
            columns_order = features_order[:self.n_top_columns]
        elif self.sort_feature_by == 'drift + importance' and features_order is not None:
            sorted_by = 'the sum of the drift score and the feature importance'
            feature_columns = [feat for feat in features_order if feat in results]
            key = lambda col: (results[col]['Drift score'] or 0) + results[col]['Importance']
            columns_order = sorted(feature_columns, key=key, reverse=True)[:self.n_top_columns]
        else:
            sorted_by = 'drift score'
            key = lambda col: results[col]['Drift score'] or 0
            columns_order = sorted(results.keys(), key=key, reverse=True)[:self.n_top_columns]

        if drift_kind == 'tabular-features':
            check_target = 'features'
            footnote = 'If available, the plot titles also show the feature importance (FI) rank'
        elif drift_kind == 'nlp-properties':
            check_target = 'properties'
            footnote = ''

        headnote = [
            textwrap.dedent(
                f"""
                <span>
                The Drift score is a measure for the difference between two distributions, in this check - the test
                and train distributions.<br> The check shows the drift score and distributions for the {check_target},
                sorted by {sorted_by} and showing only the top {self.n_top_columns} {check_target},
                according to {sorted_by}.
                </span>
                """
            ),
            get_drift_plot_sidenote(
                self.max_num_categories_for_display,
                self.show_categories_by
            ),
            footnote
        ]

        if not_enough_samples:
            headnote.append(
                '<span>The following columns do not have enough samples to calculate drift '
                f'score: {not_enough_samples}</span>'
            )

        displays = [
            *headnote,
            *(plots[col] for col in columns_order if results[col]['Drift score'] is not None)
        ]

        return results, displays

    def add_condition_drift_score_less_than(
        self: Self,
        max_allowed_categorical_score: float = 0.2,
        max_allowed_numeric_score: float = 0.2,
        allowed_num_features_exceeding_threshold: int = 0
    ) -> Self:
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            The max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.2
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
