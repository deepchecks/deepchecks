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

import pandas as pd

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data', 'load_fitted_model']
_TRAIN_DATA_URL = 'https://figshare.com/ndownloader/files/37468900'
_TEST_DATA_URL = 'https://figshare.com/ndownloader/files/37468957'
_target = 'price'
_datetime = 'datestamp'
_CAT_FEATURES = ['room_type', 'neighbourhood', 'neighbourhood_group', 'has_availability']
_NUM_FEATURES = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
                 'availability_365']


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
    usable_columns = _NUM_FEATURES + _CAT_FEATURES + [_target] + [_datetime]
    train = pd.read_csv(_TRAIN_DATA_URL, usecols=usable_columns).set_index(_datetime)
    test = pd.read_csv(_TEST_DATA_URL, usecols=usable_columns).set_index(_datetime)

    if not as_train_test:
        dataset = pd.concat([train, test], axis=0)
        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES,
                              set_datetime_from_dataframe_index=True)

        return dataset
    else:

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES, datetime_name=_datetime,
                            set_datetime_from_dataframe_index=True)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES, datetime_name=_datetime,
                           set_datetime_from_dataframe_index=True)

        return train, test


def load_fitted_model(pretrained=False):  # pylint: disable=unused-argument
    """Load and return a fitted regression model to predict the price in the Airbnb dataset.

    Returns
    -------
    model : Joblib
        the model/pipeline that was trained on the Avocado dataset.

    """
    usable_columns = [_target] + [_datetime]
    train = pd.read_csv(_TRAIN_DATA_URL, usecols=usable_columns)
    test = pd.read_csv(_TEST_DATA_URL, usecols=usable_columns)

    class AirbnbDummyModel:

        def __init__(self, all_data: pd.DataFrame):
            self._prediction_dict = all_data.set_index(_datetime)

        def predict(self, data: pd.DataFrame):
            return self._prediction_dict.loc[data.index].values

    return AirbnbDummyModel(pd.concat([train, test], axis=0))
