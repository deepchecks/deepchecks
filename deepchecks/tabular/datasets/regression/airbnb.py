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
"""Module for loading the New York City Airbnb 2019 Open Dataset.

The New York City Airbnb 2019 Open Data is a dataset containing varius details about a listed unit, when the goal
is to predict the rental price of a unit.

This dataset contains the details for units listed in NYC during 2019, was adapted from the following open kaggle
dataset: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data. This, in turn was downloaded from
the Airbnb data repository http://insideairbnb.com/get-the-data.

This dataset is licensed under the Creative Commons Attribution 4.0 International License
(https://creativecommons.org/licenses/by/4.0/).

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
import typing as t
from typing import Tuple

import numpy as np
import pandas as pd

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data', 'load_pre_calculated_prediction', 'load_pre_calculated_feature_importance']

from numpy import ndarray

_TRAIN_DATA_URL = 'https://figshare.com/ndownloader/files/37468900'
_TEST_DATA_URL = 'https://figshare.com/ndownloader/files/37468957'
_target = 'price'
_predictions = 'predictions'
_datetime = 'datestamp'
_CAT_FEATURES = ['room_type', 'neighbourhood', 'neighbourhood_group', 'has_availability']
_NUM_FEATURES = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
                 'availability_365']
_FEATURES = _NUM_FEATURES + _CAT_FEATURES


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Airbnb NYC 2019 dataset (regression).

    Parameters
    ----------
    data_format : str , default: Dataset
        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
        'Dataset' will return the data as a Dataset object
        'Dataframe' will return the data as a pandas Dataframe object
    as_train_test : bool , default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.

    Returns
    -------
    dataset : Union[deepchecks.Dataset, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train_data, test_data : Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset split to train and test sets.
    """
    train = pd.read_csv(_TRAIN_DATA_URL).drop(_predictions, axis=1)
    test = pd.read_csv(_TEST_DATA_URL).drop(_predictions, axis=1)

    if not as_train_test:
        dataset = pd.concat([train, test], axis=0)
        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES,
                              datetime_name=_datetime, features=_FEATURES)
        return dataset
    else:
        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES,
                            datetime_name=_datetime, features=_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES,
                           datetime_name=_datetime, features=_FEATURES)
        return train, test


def load_pre_calculated_prediction() -> Tuple[ndarray, ndarray]:
    """Load the pre-calculated prediction for the Airbnb NYC 2019 dataset.

    Returns
    -------
    predictions : Tuple(np.ndarray, np.ndarray)
        The first element is the pre-calculated prediction for the train set.
        The second element is the pre-calculated prediction for the test set.
    """
    usable_columns = [_datetime, _predictions]
    train = pd.read_csv(_TRAIN_DATA_URL, usecols=usable_columns)
    test = pd.read_csv(_TEST_DATA_URL, usecols=usable_columns)
    return np.asarray(train[_predictions]), np.asarray(test[_predictions])


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
