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
from typing import List, Union
import pandas as pd
import numpy as np
from PyNomaly import loop
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import (DeepchecksProcessError, DeepchecksTimeoutError, DeepchecksValueError,
                                    NotEnoughSamplesError)
from deepchecks.tabular import  SingleDatasetCheck
from deepchecks.recommender import  Context, InteractionDataset
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import  format_percent
from deepchecks.utils.typing import Hashable
import plotly.express as px

__all__ = ['ColdStartDetection']



class ColdStartDetection(SingleDatasetCheck):
    """Retrieve cold start users and items, which are new entities with limited or no historical data or interactions in the system.

    The proportion of cold start entities leads to increased uncertainty and difficulty in personalization.
    This may result in reduced user engagement, missed opportunities, and limited user understanding.
    The model is trained on the remaining historical data for each user and evaluated on the held-out interactions.
        
    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_samples : int , default: 5_000
        number of samples to use for this check.
    n_to_show : int , default: 5
        number of data elements with the highest outlier score to show (out of sample).
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_samples: int = 5_000,
            n_to_show: int = 10,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind : InteractionDataset) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.sample(self.n_samples, random_state=self.random_state)
        dataset.assert_datetime()

        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)
        user_col = dataset.user_index_name
        item_col = dataset.item_index_name

        # Group the DataFrame by user
        grouped_users = df.groupby(user_col).size()

        # Group the DataFrame by item
        grouped_items = df.groupby(item_col).size()

        cold_start_users = grouped_users[grouped_users == 1]
        cold_start_items = grouped_items[grouped_items == 1]

                    

        cold_start_users_pct = len(cold_start_users)/len(grouped_users)
        cold_start_items_pct = len(cold_start_items)/len(grouped_items)

        results_df = pd.DataFrame([['Users',100*cold_start_users_pct],
                                   ['Items',100*cold_start_items_pct]], columns=['Entity', 'Cold Start Proportion (%)'])
        
        if context.with_display:
            fig = px.bar(results_df,y='Cold Start Proportion (%)', x='Entity')
            fig.update_layout(title='Cold Start Proportion per entity (%)')
            fig.update_layout(bargap=0.2)  # Adjust the value as needed
            fig.update_traces(width=0.6)  # Adjust the value as needed
            fig.update_layout(yaxis_range=[0, 100])  # Set the range of y-axis to 0 and 100
        text = f'{format_percent(cold_start_users_pct)} of users are cold start users, {format_percent(cold_start_items_pct)} of items are cold start items.'
        return CheckResult(cold_start_users_pct, display=[text,fig])
