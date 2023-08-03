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
"""The date_leakage check module."""
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import TrainTestCheck
from deepchecks.recommender import Context
from deepchecks.utils.strings import format_datetime, format_percent
from deepchecks.tabular.utils.task_type import TaskType
import plotly.graph_objects as go
__all__ = ['DateTrainTestLeakageOverlap']


class DateTrainTestLeakageOverlap(TrainTestCheck):
    """Check test data that is dated earlier than the latest date in train. \
       The validation per user (validation_per_user=True) use the "N leave out" validation schema which is largely used in recommender systems \
        to evaluate the performance of the model by leaving out a certain number of items or interactions for each user during training. \
        In the context of validating by using the last history of each user, the leave-n-out approach involves withholding the most recent n items \
        or interactions of each user from the training data and using them as a validation set. \
        The model is trained on the remaining historical data for each user and then evaluated on the held-out interactions. \
        Whereas, when time validation is used (validation_per_user=True), we will validate in the last time window of our data.

    Parameters
    ----------
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    validation_per_user : bool, default: True
         if set to True :evaluate the performance of the model by leaving out N items for each user.
    """

    def __init__(
        self,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        validation_per_user: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.random_state = random_state
        self.validation_per_user = validation_per_user

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is the ratio of date leakage.
            data is html display of the checks' textual result.

        Raises
        ------
        DeepchecksValueError
            If one of the datasets is not a Dataset instance with an date
        """
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        display = []

        train_dataset.assert_datetime()
        train_date = train_dataset.datetime_col
        test_date = test_dataset.datetime_col

        if self.validation_per_user==False:
            _, max_train_date = min(train_date), max(train_date)
            min_test_date, max_test_date = min(test_date), max(test_date)
        
            dates_leaked = sum(date < max_train_date for date in test_date)
            if dates_leaked > 0:
                leakage_ratio = dates_leaked / test_dataset.n_samples
            else:
                leakage_ratio = 0
            fig = go.Figure()
            # Add trace for the first list
            fig.add_trace(go.Scatter(
                x=sorted(train_date.tolist()),
                y=[1] * len(train_date.tolist()),  # Use a constant y-value for all points in list1
                mode='markers',
                name='Train timestamps',
                marker=dict(
                    color='blue'
                )
            ))

            # Add trace for the second list
            fig.add_trace(go.Scatter(
                x=sorted(test_date.tolist()),
                y=[2] * len(train_date.tolist()),  # Use a different constant y-value for all points in list2
                mode='markers',
                name='Validation timestamps',
                marker=dict(
                    color='red'
                )
            ))
            
            # Add a Marker for max train date
            fig.add_trace(go.Scatter(x=[max_train_date,max_train_date], y=[0,1], name='Max Train Timestamp',
                                    line=dict(color='blue', width=2, dash='dot')))
            
            # Add a Marker for min test date
            fig.add_trace(go.Scatter(x=[min_test_date,min_test_date], y=[3,2], name='Min Test Timestamp',
                                    line=dict(color='red', width=2, dash='dot')))
            # Set the layout
            fig.update_layout(
                title='Temporal Comparison of train and validation timestamps.',
                yaxis=dict(
                    range=[0, 3],  # Adjust the range according to the number of lists
                    tickvals=[1,2],
                    ticktext=['Train','Validation']
                ),
                xaxis_title="Timestamps",
                yaxis_title="Set",
                height=400  # Adjust the height as needed
            )
                        # Show the plot
            text = f'{format_percent(leakage_ratio)} of test data samples are in the date range ' \
            f'{format_datetime(min_test_date)} - {format_datetime(max_test_date)}'\
            f', which occurs before last training data date ({format_datetime(max_train_date)})'
            display.append(text)
            display.append(fig)
            
            return_value={'max_train_date': max_train_date,
                          'min_test_date' : min_test_date
                          }    
        else:
            user_id =train_dataset.user_index_name
            train_grouped = train_dataset.data.groupby(user_id)
            test_grouped = test_dataset.data.groupby(user_id)
            leakage_ratios = []
            
            for user_id, train_group in train_grouped:
                if user_id in list(test_grouped.groups.keys()):
                    
                    train_date = train_group['timestamp']
                    _, max_train_date = min(train_date), max(train_date)

                    test_group = test_grouped.get_group(user_id)
                    test_date = test_group['timestamp']
 
                    dates_leaked = sum(date < max_train_date for date in test_date)
                    if dates_leaked > 0:
                        leakage_ratio = dates_leaked / len(test_group)
                        leakage_ratios.append(leakage_ratio)
                else:
                    continue
            if len(leakage_ratios) > 0:
                average_leakage_ratio = sum(leakage_ratios) / len(leakage_ratios)
                text = f'There is an average leak of {format_percent(average_leakage_ratio)} per user. In other words, {format_percent(average_leakage_ratio)} of the  validation set of a user, appears before the maximum timestamp of the training set.'
                display.append(text)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=['Leakage Percentage'], y=[100*average_leakage_ratio],width=0.3))

                fig.update_layout(
                    title_text="Leaky User Overview",
                    yaxis_title="Proportion (%)"
                    )

                display.append(fig)
            else:
                average_leakage_ratio = 0
                display = None   

            # Display the figure
            return_value={'average_leakage_ratio' : average_leakage_ratio}   
        
        return CheckResult(value=return_value, header='Date Train-Test Leakage (overlap)', display=display)
