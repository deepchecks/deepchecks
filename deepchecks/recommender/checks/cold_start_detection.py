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
"""Outlier detection functions."""
import time
from typing import List, Union

import numpy as np
from PyNomaly import loop
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import (DeepchecksProcessError, DeepchecksTimeoutError, DeepchecksValueError,
                                    NotEnoughSamplesError)
from deepchecks.tabular import  SingleDatasetCheck
from deepchecks.recommender import  Context

from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['ColdStartDetection']

DATASET_TIME_EVALUATION_SIZE = 100
MINIMUM_NUM_NEAREST_NEIGHBORS = 5


class ColdStartDetection(SingleDatasetCheck):
    """Retrieve cold start users.

    The LoOP algorithm is a robust method for detecting outliers in a dataset across multiple variables by comparing
    the density in the area of a sample with the densities in the areas of its nearest neighbors.
    The output of the algorithm is highly dependent on the number of nearest neighbors, it is recommended to
    select a value k that represent the maximum cluster size that will still be considered as "outliers".
    See https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf for more details.
    LoOP relies on a distance matrix, in our implementation we use the Gower distance that measure the distance
    between two samples based on its numeric and categorical features.
    See https://statisticaloddsandends.wordpress.com/2021/02/23/what-is-gowers-distance/ for further details.

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
    timeout : int, default: 10
        Check will be interrupted if it takes more than this number of seconds. If 0, check will not be interrupted.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            n_samples: int = 5_000,
            n_to_show: int = 10,
            random_state: int = 42,
            timeout: int = 10,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        #dataset = dataset.sample(self.n_samples, random_state=self.random_state).drop_na_labels()
        dataset.assert_datetime()

        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)
        datetime_col = dataset.datetime_name
        user_col = dataset.user_index_name
        item_col = dataset.item_index_name

        df = df.sort_values(datetime_col)

        # Group the DataFrame by user
        grouped_df = df.groupby(user_col)

        # Calculate session length (number of interactions) for each user
        session_lengths = grouped_df.size()

        # Find the last interaction timestamp for each user
        last_interaction_timestamps = grouped_df[datetime_col].last()
        last_interaction_item= grouped_df[item_col].last()

        # Create a new DataFrame with user, session length, and last interaction timestamp
        extracted_df = pd.DataFrame({
            f'{user_col}': last_interaction_timestamps.index,
            'session_length': session_lengths,
            f'{item_col}': last_interaction_item,
            f'{datetime_col}': last_interaction_timestamps
        })
        
        cold_start_users = extracted_df[extracted_df['session_length'] == 1]
        cold_start_users_to_show = cold_start_users.sample(n=self.n_to_show, random_state=42)
        # Create the check result visualization
        headnote = """<span>
                    cold start users refer to new users who have limited or no historical data or interactions within a system or platform.
                    They have not yet established a significant user history that can be leveraged for personalized recommendations or targeted actions.
                    For the model, there can be some consequences fromincreased uncertainty to the difficulty in personalization.
                    From the business side, the consequences are reeduced user engagement, missed opportunities, and limited user understanding.
                    .<br><br>
                    </span>"""
                    

        proportion_cold_start_users = cold_start_users[user_col].nunique()/df[user_col].nunique()

        text = f'{format_percent(proportion_cold_start_users)}% of users are cold start users. '
    
        return CheckResult(cold_start_users_to_show[f'{datetime_col}'], display=[text,headnote, cold_start_users_to_show])


