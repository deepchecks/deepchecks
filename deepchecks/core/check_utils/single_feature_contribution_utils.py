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
"""Module containing common SingleFeatureContribution (PPS) utils."""
from typing import Optional

import numpy as np
import pandas as pd
import deepchecks.ppscore as pps
from deepchecks.utils.plot import colors
from deepchecks.utils.typing import Hashable
import plotly.graph_objects as go


def get_pps_figure(per_class: bool):
    """If per_class is True, then no title is defined on the figure."""
    fig = go.Figure()
    fig.update_layout(
        yaxis_title='Predictive Power Score (PPS)',
        yaxis_range=[0, 1.05],
        legend=dict(x=1.0, y=1.0),
        barmode='group',
        width=800, height=500
    )
    if per_class:
        fig.update_layout(xaxis_title='Class')
    else:
        fig.update_layout(
            title='Predictive Power Score (PPS) - Can a feature predict the label by itself?',
            xaxis_title='Column',
        )
    return fig


def pps_df_to_trace(s_pps: pd.Series, name: str):
    """If name is train/test use our defined colors, else will use plotly defaults."""
    name = name.capitalize() if name else None
    return go.Bar(x=s_pps.index,
                  y=s_pps,
                  name=name,
                  marker_color=colors.get(name),
                  text=s_pps.round(2),
                  textposition='outside'
                  )


def get_single_feature_contribution(train_df: pd.DataFrame, train_label_name: Optional[Hashable], test_df: pd.DataFrame,
                                    test_label_name: Optional[Hashable], ppscore_params: dict, n_show_top: int,
                                    random_state: int = None):
    """
    Calculate the PPS for train, test and difference for single feature contribution checks.

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

    s_difference_to_display = np.abs(s_difference).apply(lambda x: 0 if x < 0 else x)
    s_difference_to_display = s_difference_to_display.sort_values(ascending=False).head(n_show_top)

    s_pps_train_to_display = s_pps_train[s_difference_to_display.index]
    s_pps_test_to_display = s_pps_test[s_difference_to_display.index]

    fig = get_pps_figure(per_class=False)
    fig.add_trace(pps_df_to_trace(s_pps_train_to_display, 'train'))
    fig.add_trace(pps_df_to_trace(s_pps_test_to_display, 'test'))
    fig.add_trace(go.Scatter(x=s_difference_to_display.index,
                             y=s_difference_to_display,
                             name='Train-Test Difference (abs)',
                             marker=dict(symbol='circle', size=15),
                             line=dict(color='#aa57b5', width=5)
                             ))

    ret_value = {'train': s_pps_train.to_dict(), 'test': s_pps_test.to_dict(),
                 'train-test difference': s_difference.to_dict()}

    # display only if not all scores are 0
    display = [fig] if s_pps_train.sum() or s_pps_test.sum() else None

    return ret_value, display


def get_single_feature_contribution_per_class(train_df: pd.DataFrame, train_label_name: Optional[Hashable],
                                              test_df: pd.DataFrame,
                                              test_label_name: Optional[Hashable], ppscore_params: dict,
                                              n_show_top: int,
                                              min_pps_to_show: float = 0.05,
                                              random_state: int = None):
    """
    Calculate the PPS for train, test and difference for single feature contribution checks per class.

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

        # If not all results are 0, plot add to display:
        if any(s_train > min_pps_to_show) or any(s_test > min_pps_to_show):
            s_difference_to_display = np.abs(s_difference).apply(lambda x: 0 if x < 0 else x)
            s_difference_to_display = s_difference_to_display.sort_values(ascending=False).head(n_show_top)

            s_train_to_display = s_train[s_difference_to_display.index]
            s_test_to_display = s_test[s_difference_to_display.index]

            fig = get_pps_figure(per_class=True)
            fig.update_layout(title=f'{feature}: Predictive Power Score (PPS) Per Class')
            fig.add_trace(pps_df_to_trace(s_train_to_display, 'train'))
            fig.add_trace(pps_df_to_trace(s_test_to_display, 'test'))
            display.append(fig)

    return ret_value, display
