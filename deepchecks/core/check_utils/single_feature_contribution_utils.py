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
import plotly.graph_objects as go

from deepchecks.utils.typing import Hashable


def get_single_feature_contribution(train_df: pd.DataFrame, train_label_name: Optional[Hashable], test_df: pd.DataFrame,
                                    test_label_name: Optional[Hashable], ppscore_params: dict, n_show_top: int):
    df_pps_train = pps.predictors(df=train_df, y=train_label_name,
                                  random_seed=42,
                                  **ppscore_params)
    df_pps_test = pps.predictors(df=test_df,
                                 y=test_label_name,
                                 random_seed=42, **ppscore_params)

    s_pps_train = df_pps_train.set_index('x', drop=True)['ppscore']
    s_pps_test = df_pps_test.set_index('x', drop=True)['ppscore']
    s_difference = s_pps_train - s_pps_test

    s_difference_to_display = np.abs(s_difference).apply(lambda x: 0 if x < 0 else x)
    s_difference_to_display = s_difference_to_display.sort_values(ascending=False).head(n_show_top)

    s_pps_train_to_display = s_pps_train[s_difference_to_display.index]
    s_pps_test_to_display = s_pps_test[s_difference_to_display.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=s_pps_train_to_display.index,
                         y=s_pps_train_to_display,
                         name='Train',
                         marker_color=colors['Train'], text=s_pps_train_to_display.round(2), textposition='outside'
                         ))
    fig.add_trace(go.Bar(x=s_pps_test_to_display.index,
                         y=s_pps_test_to_display,
                         name='Test',
                         marker_color=colors['Test'], text=s_pps_test_to_display.round(2), textposition='outside'
                         ))
    fig.add_trace(go.Scatter(x=s_difference_to_display.index,
                             y=s_difference_to_display,
                             name='Train-Test Difference (abs)',
                             marker=dict(symbol='circle', size=15),
                             line=dict(color='#aa57b5', width=5)
                             ))

    fig.update_layout(
        title='Predictive Power Score (PPS) - Can a feature predict the label by itself?',
        xaxis_title='Column',
        yaxis_title='Predictive Power Score (PPS)',
        yaxis_range=[0, 1.05],
        legend=dict(x=1.0, y=1.0),
        barmode='group',
        width=800, height=500
    )

    ret_value = {'train': s_pps_train.to_dict(), 'test': s_pps_test.to_dict(),
                 'train-test difference': s_difference.to_dict()}

    # display only if not all scores are 0
    display = [fig] if s_pps_train.sum() or s_pps_test.sum() else None

    return ret_value, display
