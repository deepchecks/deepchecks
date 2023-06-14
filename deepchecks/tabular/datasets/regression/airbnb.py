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
"""Module for loading the New York City Airbnb 2019 Open Dataset.

The New York City Airbnb 2019 Open Data is a dataset containing varius details about a listed unit, when the goal
is to predict the rental price of a unit.

This dataset contains the details for units listed in NYC during 2019, was adapted from the following open kaggle
dataset: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data. This, in turn was downloaded from
the Airbnb data repository http://insideairbnb.com/get-the-data.

This dataset is licensed under the CC0 1.0 Universal License (https://creativecommons.org/publicdomain/zero/1.0/).

The typical ML task in this dataset is to build a model that predicts the average rental price of a unit.

Dataset Shape:
    .. list-table:: Dataset Shape
       :widths: 50 50
       :header-rows: 1

       * - Property
         - Value
       * - Samples Total
         - 47.3K
       * - Dimensionality
         - 9
       * - Features
         - real, string
       * - Targets
         - int 31 - 795

Description:
    .. list-table:: Dataset Description
       :widths: 50 50 50
       :header-rows: 1

       * - Column name
         - Column Role
         - Description
       * - datestamp
         - Datetime
         - The date of the observation
       * - neighbourhood_group
         - Feature
         -
       * - neighbourhood
         - Feature
         -
       * - room_type
         - Feature
         -
       * - minimum_nights
         - Feature
         -
       * - number_of_reviews
         - Feature
         -
       * - reviews_per_month
         - Feature
         -
       * - calculated_host_listings_count
         - Feature
         -
       * - availability_365
         - Feature
         -
       * - has_availability
         - Feature
         -
       * - price
         - Label
         - The rental price of the unit
"""
import math
import time
import typing as t

import numpy as np
import pandas as pd

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data_and_predictions', 'load_pre_calculated_feature_importance']

_TRAIN_DATA_URL = 'https://drive.google.com/uc?export=download&id=1UWkr1BQlyyUkbsW5hHIFTr-x0evZE3Ie'
_TEST_DATA_URL = 'https://drive.google.com/uc?export=download&id=1lfpWVtDktrnsLUzCN1tkRc1jRbguEz3a'
_target = 'price'
_predictions = 'predictions'
_datetime = 'timestamp'
_CAT_FEATURES = ['room_type', 'neighbourhood', 'neighbourhood_group', 'has_availability']
_NUM_FEATURES = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
                 'availability_365']
_FEATURES = _NUM_FEATURES + _CAT_FEATURES


def load_data_and_predictions(data_format: str = 'Dataset', load_train: bool = True, modify_timestamps: bool = True,
                              data_size: t.Optional[int] = 15000, random_state: int = 42) \
        -> t.Tuple[t.Union[Dataset, pd.DataFrame], np.ndarray]:
    """Load and returns the Airbnb NYC 2019 dataset (regression).

    Parameters
    ----------
    data_format : str , default: Dataset
        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
        'Dataset' will return the data as a Dataset object
        'Dataframe' will return the data as a pandas Dataframe object
    load_train : bool , default: True
        If True, the returned data is the train data. otherwise the test dataset.
    modify_timestamps : bool , default: True
        If True, the returned data timestamp column will be for the last 30 days.
        Otherwise, the data timestamp will be for March 2023.
    data_size : t.Optional[int] , default: 15000
        The number of samples to return. If None, returns all the data.
    random_state : int , default 42
        The random state to use for sampling.
    Returns
    -------
    dataset, predictions : Tuple[Union[deepchecks.Dataset, pd.DataFrame], np.ndarray]
        Tuple of the deepchecks dataset or dataframe and the predictions.
    """
    if load_train:
        dataset = pd.read_csv(_TRAIN_DATA_URL)
    else:
        dataset = pd.read_csv(_TEST_DATA_URL)

    if data_size is not None:
        if data_size < len(dataset):
            dataset = dataset.sample(data_size, random_state=random_state)
        elif data_size > len(dataset):
            dataset = pd.concat([dataset] * math.ceil(data_size / len(dataset)), axis=0, ignore_index=True)
            dataset = dataset.sample(data_size, random_state=random_state)
            if not load_train:
                dataset = dataset.sort_values(_datetime)

    if modify_timestamps and not load_train:
        current_time = int(time.time())
        time_test_start = current_time - 86400 * 30  # Span data for 30 days
        dataset[_datetime] = np.sort(
            (np.random.rand(len(dataset)) * (current_time - time_test_start)) + time_test_start
        )
        dataset[_datetime] = dataset[_datetime].apply(lambda x: pd.Timestamp(x, unit='s'))

    predictions = np.asarray(dataset[_predictions])
    dataset.drop(_predictions, axis=1, inplace=True)
    if data_format == 'Dataset':
        dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES,
                          features=_FEATURES)
    return dataset, predictions


def load_pre_calculated_feature_importance() -> pd.Series:
    """Load the pre-calculated feature importance for the Airbnb NYC 2019 dataset.

    Returns
    -------
    feature_importance : pd.Series
        The feature importance for a model trained on the Airbnb NYC 2019 dataset.
    """
    return pd.Series({
        'neighbourhood_group': 0.1,
        'neighbourhood': 0.2,
        'room_type': 0.1,
        'minimum_nights': 0.1,
        'number_of_reviews': 0.1,
        'reviews_per_month': 0.1,
        'calculated_host_listings_count': 0.1,
        'availability_365': 0.1,
        'has_availability': 0.1,
    })
