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
"""Module of DateTrainTestLeakageOverlap check."""
from deepchecks.core import CheckResult
from deepchecks.tabular import TrainTestCheck
from deepchecks.recommender import Context
from deepchecks.utils.strings import format_datetime, format_percent
import numpy as np
import pandas as pd
import plotly.graph_objects as go
__all__ = ['DateTrainTestLeakageOverlap']


class DateTrainTestLeakageOverlap(TrainTestCheck):
    """Ensure there's no overlap between training and testing data based on dates.

    Time validation (validation_per_user=True) is used to validate in the last time window.
    For validation per user (validation_per_user=True), use "N leave out" validation schema.
    It involves withholding the most recent n interactions for each user for validation.

    Parameters
    ----------
    validation_per_user : bool, default: True
         if set to True :evaluate the performance of the model by leaving out N items for each user.
    """

    def __init__(
        self,
        validation_per_user: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
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
        train_date = context.train.datetime_col
        test_date = context.test.datetime_col
        display = []
        if self.validation_per_user is False:
            _, max_train_date = min(train_date), max(train_date)
            min_test_date, max_test_date = min(test_date), max(test_date)

            dates_leaked = sum(date < max_train_date for date in test_date)
            if dates_leaked > 0:
                leakage_ratio = dates_leaked / context.test.n_samples
            else:
                leakage_ratio = 0

            # Optimize the plotting by sampling data
            sample_size = min(len(train_date), len(test_date), 100_000)
            train_sample = np.random.choice(train_date, sample_size, replace=False)
            test_sample = np.random.choice(test_date, sample_size, replace=False)
            # Ensure min and max dates of each set are included in the sample
            train_sample = np.append(train_sample,
                                     pd.to_datetime([min(train_date), max(train_date)]))
            test_sample = np.append(test_sample,
                                    pd.to_datetime([min(test_date), max(test_date)]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sorted(train_sample.tolist())),
                y=[1] * len(train_sample),
                mode='lines',
                name='Train timestamps',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sorted(test_sample.tolist())),
                y=[2] * len(test_sample),
                mode='lines',
                name='Validation timestamps',
                line=dict(color='red')
            ))
            # Add Markers for max_train_date and min_test_date
            fig.add_trace(go.Scatter(
                x=[max(train_date), max(train_date)], y=[0, 1],
                name='Max Train Timestamp',
                line=dict(color='blue', width=2, dash='dot'))
            )

            fig.add_trace(go.Scatter(
                x=[min(test_date), min(test_date)], y=[3, 2],
                name='Min Test Timestamp',
                line=dict(color='red', width=2, dash='dot'))
            )

            fig.update_layout(
                title='Temporal Comparison of train and validation timestamps.',
                yaxis=dict(
                    range=[0, 3],
                    tickvals=[1, 2],
                    ticktext=['Train', 'Validation']
                ),
                xaxis=dict(
                    title='Timestamps',
                    type='date',  # Set the x-axis type to 'date' for proper date formatting
                    tickformat='%Y-%m-%d',  # Customize the date format as needed
                ),
                xaxis_title='Timestamps',
                yaxis_title='Set',
                height=400
            )

            text = f'{format_percent(leakage_ratio)} of test data samples are in the date range '\
                f'{format_datetime(min_test_date)} - {format_datetime(max_test_date)}'\
                f', which occurs before last training data date ({format_datetime(max_train_date)})'
            display.append(text)
            return_value = {
                          'max_train_date': max_train_date,
                          'min_test_date': min_test_date
                            }
            display.append(fig)
        else:
            user_id = context.train.user_index_name
            train_grouped = context.train.data.groupby(user_id)
            test_grouped = context.test.data.groupby(user_id)
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
                text = f'There is an average leak of {format_percent(average_leakage_ratio)} per user.\
                    In other words, {format_percent(average_leakage_ratio)} of the  validation set of a user\
                    appears before the maximum timestamp of the training set.'
                display.append(text)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=['Leakage Percentage'], y=[100*average_leakage_ratio], width=0.3))

                fig.update_layout(
                    title_text='Leaky User Overview',
                    yaxis_title='Proportion (%)'
                    )

                display.append(fig)
            else:
                average_leakage_ratio = 0
                text = 'No leak between users.'
                display.append(text)

            return_value = {'average_leakage_ratio' : average_leakage_ratio}

        return CheckResult(value=return_value,
                           header='Date Train-Test Leakage (overlap)',
                           display=display)
