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
"""Module containing the train test performance check."""
import abc
import typing as t

import pandas as pd
import plotly.express as px
from typing_extensions import Self

from deepchecks.core.check_utils.class_performance_utils import (
    get_condition_class_performance_imbalance_ratio_less_than, get_condition_test_performance_greater_than,
    get_condition_train_test_relative_degradation_less_than)
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors
from deepchecks.utils.strings import format_percent

__all__ = ['TrainTestPerformanceAbstract']


class TrainTestPerformanceAbstract(abc.ABC):
    """Base functionality for some train-test performance checks."""

    add_condition: t.Callable[..., t.Any]

    def _prepare_display(
            self,
            results: pd.DataFrame,
            train_dataset_name: str,
            test_dataset_name: str,
            classes_without_enough_samples: t.Optional[t.List[str]] = None,
            top_classes_to_show: t.Optional[t.List[str]] = None
    ):
        display_df = results.replace({
            'Dataset': {
                DEFAULT_DATASET_NAMES[0]: train_dataset_name,
                DEFAULT_DATASET_NAMES[1]: test_dataset_name
            }
        })

        figures = []
        data_scorers_per_class = display_df[results['Class'].notna()]
        data_scorers_per_dataset = display_df[results['Class'].isna()].drop(columns=['Class'])

        # Filter classes without enough samples and get display comment for them:
        if classes_without_enough_samples:
            data_scorers_per_class = \
                data_scorers_per_class.loc[~data_scorers_per_class['Class'].isin(classes_without_enough_samples)]

        # Filter top classes to show:
        if top_classes_to_show:
            not_shown_classes = list(set(data_scorers_per_class['Class'].unique()) - set(top_classes_to_show))
            data_scorers_per_class = \
                data_scorers_per_class.loc[data_scorers_per_class['Class'].isin(top_classes_to_show)]
        else:
            not_shown_classes = None

        for data in (data_scorers_per_dataset, data_scorers_per_class):
            if data.shape[0] == 0:
                continue

            fig = px.histogram(
                data,
                x='Class' if 'Class' in data.columns else 'Dataset',
                y='Value',
                color='Dataset',
                barmode='group',
                facet_col='Metric',
                facet_col_spacing=0.05,
                hover_data=['Number of samples'],
                color_discrete_map={
                    train_dataset_name: colors[DEFAULT_DATASET_NAMES[0]],
                    test_dataset_name: colors[DEFAULT_DATASET_NAMES[1]]
                },
            )

            figures.append(
                fig.update_xaxes(
                    title=None,
                    type='category',
                    tickangle=60,
                )
                .update_yaxes(title=None, matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
                .add_annotation(
                    text='Class',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    y=-0.1,
                    x=-0.1
                )
            )

        # Add comments about not shown classes:
        df = pd.DataFrame({}, columns=['Reason', 'Classes']).set_index('Reason')
        if not_shown_classes:
            df.loc[f'Not shown classes (showing only top {len(top_classes_to_show)})'] = str(not_shown_classes)
        if classes_without_enough_samples:
            df.loc[f'Classes without enough samples in either {train_dataset_name} or {test_dataset_name}'] = \
                str(classes_without_enough_samples)

        if not df.empty:
            figures.append(df)

        return figures

    def add_condition_test_performance_greater_than(self: Self, min_score: float) -> Self:
        """Add condition - metric scores are greater than the threshold.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        condition = get_condition_test_performance_greater_than(min_score=min_score)
        return self.add_condition(f'Scores are greater than {min_score}', condition)

    def add_condition_train_test_relative_degradation_less_than(self: Self, threshold: float = 0.1) -> Self:
        """Add condition - test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float , default: 0.1
            maximum degradation ratio allowed (value between 0 and 1)
        """
        name = f'Train-Test scores relative degradation is less than {threshold}'
        condition = get_condition_train_test_relative_degradation_less_than(threshold=threshold)
        return self.add_condition(name, condition)

    def add_condition_class_performance_imbalance_ratio_less_than(
            self: Self,
            score: str,
            threshold: float = 0.3,
    ) -> Self:
        """Add condition - relative ratio difference between highest-class and lowest-class is less than threshold.

        Parameters
        ----------
        threshold : float , default: 0.3
            ratio difference threshold
        score : str
            limit score for condition

        Returns
        -------
        Self
            instance of 'TrainTestPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed.
        """
        name = f"Relative ratio difference between labels '{score}' score is less than {format_percent(threshold)}"
        condition = get_condition_class_performance_imbalance_ratio_less_than(threshold=threshold, score=score)
        return self.add_condition(name=name, condition_func=condition)
