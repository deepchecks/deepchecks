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
"""Module containing common feature label correlation (PPS) utils."""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import deepchecks.ppscore as pps
from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable


def get_pps_figure(per_class: bool, n_of_features: int):
    """If per_class is True, then no title is defined on the figure."""
    fig = go.Figure()
    fig.update_layout(
        yaxis_title='Predictive Power Score (PPS)',
        yaxis_range=(0, 1.05),
        # NOTE:
        # the range, in this case, is needed to fix a problem with
        # too wide bars when there are only one or two of them`s on
        # the plot, plus it also centralizes them`s on the plot
        # The min value of the range (range(min. max)) is bigger because
        # otherwise bars will not be centralized on the plot, they will
        # appear on the left part of the plot (that is probably because of zero)
        xaxis_range=(-3, n_of_features + 2),
        legend=dict(x=1.0, y=1.0),
        barmode='group',
        height=500,
        # Set the x-axis as category, since if the column names are numbers it will infer the x-axis as numerical
        # and will show the values very far from each other
        xaxis_type='category'
    )
    if per_class:
        fig.update_layout(xaxis_title='Class')
    else:
        fig.update_layout(
            title='Predictive Power Score (PPS) - Can a feature predict the label by itself?',
            xaxis_title='Column',
        )
    return fig


def pd_series_to_trace(s_pps: pd.Series, name: str):
    """Create bar plotly bar trace out of pandas Series."""
    name = name.capitalize() if name else None
    return go.Bar(x=s_pps.index,
                  y=s_pps,
                  name=name,
                  marker_color=colors.get(name),
                  text='<b>' + s_pps.round(2).astype(str) + '</b>',
                  textposition='outside'
                  )


def pd_series_to_trace_with_diff(s_pps: pd.Series, name: str, diffs: pd.Series):
    """Create bar plotly bar trace out of pandas Series, with difference shown in percentages."""
    diffs_text = '(' + diffs.apply(format_percent, floating_point=0, add_positive_prefix=True) + ')'
    text = diffs_text + '<br>' + s_pps.round(2).astype(str)
    name = name.capitalize() if name else None
    return go.Bar(x=s_pps.index,
                  y=s_pps,
                  name=name,
                  marker_color=colors.get(name),
                  text='<b>' + text + '</b>',
                  textposition='outside'
                  )


def get_feature_label_correlation(train_df: pd.DataFrame, train_label_name: Optional[Hashable],
                                  test_df: pd.DataFrame,
                                  test_label_name: Optional[Hashable], ppscore_params: dict,
                                  n_show_top: int,
                                  min_pps_to_show: float = 0.05,
                                  random_state: int = None,
                                  with_display: bool = True):
    """
    Calculate the PPS for train, test and difference for feature label correlation checks.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    This function calculates the PPS per feature for both train and test, and returns the data and display graph.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        train_df: pd.DataFrame
            DataFrame of all train features and label
        train_label_name:: str
            name of label column in train dataframe
        test_df:
            DataFrame of all test features and label
        test_label_name: str
            name of label column in test dataframe
        ppscore_params: dict
            dictionary of additional parameters for the ppscore predictor function
        n_show_top: int
            Number of features to show, sorted by the magnitude of difference in PPS
        min_pps_to_show: float, default 0.05
            Minimum PPS to show a class in the graph
        random_state: int, default None
            Random state for the ppscore.predictors function

    Returns:
        CheckResult
            value: dictionaries of PPS values for train, test and train-test difference.
            display: bar graph of the PPS of each feature.
    """
    df_pps_train = pps.predictors(df=train_df, y=train_label_name,
                                  random_seed=random_state,
                                  **ppscore_params)
    df_pps_test = pps.predictors(df=test_df,
                                 y=test_label_name,
                                 random_seed=random_state, **ppscore_params)

    s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
    s_pps_test = df_pps_test.set_index('x', drop=True)['ppscore']
    s_difference = s_pps_train - s_pps_test

    ret_value = {'train': s_pps_train.to_dict(), 'test': s_pps_test.to_dict(),
                 'train-test difference': s_difference.to_dict()}

    if not with_display:
        return ret_value, None

    sorted_order_for_display = np.abs(s_difference).sort_values(ascending=False).head(n_show_top).index
    s_pps_train_to_display = s_pps_train[sorted_order_for_display]
    s_pps_test_to_display = s_pps_test[sorted_order_for_display]
    s_difference_to_display = s_difference[sorted_order_for_display]

    fig = get_pps_figure(per_class=False, n_of_features=len(sorted_order_for_display))
    fig.add_trace(pd_series_to_trace(s_pps_train_to_display, 'train'))
    fig.add_trace(pd_series_to_trace_with_diff(s_pps_test_to_display, 'test', -s_difference_to_display))

    # display only if not all scores are above min_pps_to_show
    display = [fig] if any(s_pps_train > min_pps_to_show) or any(s_pps_test > min_pps_to_show) else None

    return ret_value, display


def get_feature_label_correlation_per_class(train_df: pd.DataFrame, train_label_name: Optional[Hashable],
                                            test_df: pd.DataFrame,
                                            test_label_name: Optional[Hashable], ppscore_params: dict,
                                            n_show_top: int,
                                            min_pps_to_show: float = 0.05,
                                            random_state: int = None,
                                            with_display: bool = True):
    """
    Calculate the PPS for train, test and difference for feature label correlation checks per class.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    This function calculates the PPS per feature for both train and test, and returns the data and display graph.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Args:
        train_df: pd.DataFrame
            DataFrame of all train features and label
        train_label_name:: str
            name of label column in train dataframe
        test_df:
            DataFrame of all test features and label
        test_label_name: str
            name of label column in test dataframe
        ppscore_params: dict
            dictionary of additional parameters for the ppscore predictor function
        n_show_top: int
            Number of features to show, sorted by the magnitude of difference in PPS
        min_pps_to_show: float, default 0.05
            Minimum PPS to show a class in the graph
        random_state: int, default None
            Random state for the ppscore.predictors function

    Returns:
        CheckResult
            value: dictionaries of features, each value is 3 dictionaries of PPS values for train, test and
            train-test difference.
            display: bar graphs of the PPS for each feature.
    """
    df_pps_train_all = pd.DataFrame()
    df_pps_test_all = pd.DataFrame()
    df_pps_difference_all = pd.DataFrame()
    display = []
    ret_value = {}

    for c in train_df[train_label_name].unique():
        train_df_all_vs_one = train_df.copy()
        test_df_all_vs_one = test_df.copy()

        train_df_all_vs_one[train_label_name] = train_df_all_vs_one[train_label_name].apply(
            lambda x: 1 if x == c else 0)  # pylint: disable=cell-var-from-loop
        test_df_all_vs_one[test_label_name] = test_df_all_vs_one[test_label_name].apply(
            lambda x: 1 if x == c else 0)  # pylint: disable=cell-var-from-loop

        df_pps_train = pps.predictors(df=train_df_all_vs_one, y=train_label_name,
                                      random_seed=random_state,
                                      **ppscore_params)
        df_pps_test = pps.predictors(df=test_df_all_vs_one,
                                     y=test_label_name,
                                     random_seed=random_state, **ppscore_params)

        s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
        s_pps_test = df_pps_test.set_index('x', drop=True)['ppscore']
        s_difference = s_pps_train - s_pps_test

        df_pps_train_all[c] = s_pps_train
        df_pps_test_all[c] = s_pps_test
        df_pps_difference_all[c] = s_difference

    for feature in df_pps_train_all.index:
        s_train = df_pps_train_all.loc[feature]
        s_test = df_pps_test_all.loc[feature]
        s_difference = df_pps_difference_all.loc[feature]

        ret_value[feature] = {'train': s_train.to_dict(), 'test': s_test.to_dict(),
                              'train-test difference': s_difference.to_dict()}

        # display only if not all scores are above min_pps_to_show
        if with_display and any(s_train > min_pps_to_show) or any(s_test > min_pps_to_show):
            sorted_order_for_display = np.abs(s_difference).sort_values(ascending=False).head(n_show_top).index

            s_train_to_display = s_train[sorted_order_for_display]
            s_test_to_display = s_test[sorted_order_for_display]
            s_difference_to_display = s_difference[sorted_order_for_display]

            fig = get_pps_figure(per_class=True, n_of_features=len(sorted_order_for_display))
            fig.update_layout(title=f'{feature}: Predictive Power Score (PPS) Per Class')
            fig.add_trace(pd_series_to_trace(s_train_to_display, 'train'))
            fig.add_trace(pd_series_to_trace_with_diff(s_test_to_display, 'test', -s_difference_to_display))
            display.append(fig)

    return ret_value, display
