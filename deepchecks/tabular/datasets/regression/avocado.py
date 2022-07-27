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
"""The avocado dataset contains historical data on avocado prices and sales volume in multiple US markets.

The avocado dataset contains historical data on avocado prices and sales volume in multiple US markets
https://www.kaggle.com/neuromusic/avocado-prices.

This dataset is licensed under the Open Data Commons Open Database License (ODbL) v1.0
(https://opendatacommons.org/licenses/odbl/1-0/).

The typical ML task in this dataset is to build a model that predicts the average price of Avocados.

Dataset Shape:
    .. list-table:: Dataset Shape
       :widths: 50 50
       :header-rows: 1

       * - Property
         - Value
       * - Samples Total
         - 18.2K
       * - Dimensionality
         - 14
       * - Features
         - real, string
       * - Targets
         - real 0.44 - 3.25

Description:
    .. list-table:: Dataset Description
       :widths: 50 50 50
       :header-rows: 1

       * - Column name
         - Column Role
         - Description
       * - Date
         - Datetime
         - The date of the observation
       * - Total Volume
         - Feature
         - Total number of avocados sold
       * - 4046
         - Feature
         - Total number of avocados with PLU 4046 (small avocados) sold
       * - 4225
         - Feature
         - Total number of avocados with PLU 4225 (large avocados) sold
       * - 4770
         - Feature
         - Total number of avocados with PLU 4770 (xlarge avocados) sold
       * - Total Bags
         - Feature
         -
       * - Small Bags
         - Feature
         -
       * - Large Bags
         - Feature
         -
       * - XLarge Bags
         - Feature
         -
       * - type
         - Feature
         - Conventional or organic
       * - year
         - Feature
         -
       * - region
         - Feature
         - The city or region of the observation
       * - AveragePrice
         - Label
         - The average price of a single avocado
"""
import typing as t
from urllib.request import urlopen

import joblib
import pandas as pd
import sklearn
from category_encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data', 'load_fitted_model']
_MODEL_URL = 'https://figshare.com/ndownloader/files/35259829'
_FULL_DATA_URL = 'https://figshare.com/ndownloader/files/35259799'
_TRAIN_DATA_URL = 'https://figshare.com/ndownloader/files/35259769'
_TEST_DATA_URL = 'https://figshare.com/ndownloader/files/35259814'
_MODEL_VERSION = '1.0.2'
_target = 'AveragePrice'
_CAT_FEATURES = ['region', 'type']
_NUM_FEATURES = ['Total Volume', '4046', '4225', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Avocado dataset (regression).

    Parameters
    ----------
    data_format : str , default: Dataset
        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
        'Dataset' will return the data as a Dataset object
        'Dataframe' will return the data as a pandas Dataframe object
    as_train_test : bool , default: True
        If True, the returned data is splitted into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.

    Returns
    -------
    dataset : Union[deepchecks.Dataset, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train_data, test_data : Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.
    """
    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL)

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')

        return dataset
    else:
        train = pd.read_csv(_TRAIN_DATA_URL)
        test = pd.read_csv(_TEST_DATA_URL)

        if data_format == 'Dataset':
            train = Dataset(train, label='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')
            test = Dataset(test, label='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')

        return train, test


def load_fitted_model(pretrained=True):
    """Load and return a fitted regression model to predict the AveragePrice in the avocado dataset.

    Returns
    -------
    model : Joblib
        the model/pipeline that was trained on the Avocado dataset.

    """
    if sklearn.__version__ == _MODEL_VERSION and pretrained:
        with urlopen(_MODEL_URL) as f:
            model = joblib.load(f)
    else:
        model = _build_model()
        train, _ = load_data()
        model.fit(train.data[train.features], train.data[train.label_name])
    return model


def _build_model():
    """Build the model to fit."""
    return Pipeline(steps=[
        ('preprocessor',
         ColumnTransformer(transformers=[('num',
                                          Pipeline(steps=[('imputer',
                                                           SimpleImputer(strategy='median')),
                                                          ('scaler',
                                                           StandardScaler())]),
                                          _NUM_FEATURES),
                                         ('cat', OneHotEncoder(),
                                          _CAT_FEATURES)])),
        ('classifier', RandomForestRegressor(random_state=0, max_depth=7, n_estimators=30))
    ])
