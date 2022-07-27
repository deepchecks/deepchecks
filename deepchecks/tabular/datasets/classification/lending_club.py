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
"""The data set contains features for binary prediction of whether a loan will be approved or not.

The partial data has 21668 records with 26 features and one binary target column, referring to whether the specified
loan was approved. The partial data set contains the records from the years 2017,2018 for the months of June, July
and August. The train set are the records from 2017 and the test set consists of the records from 2018.

This is a partial copy of the dataset supplied in: https://www.kaggle.com/datasets/wordsforthewise/lending-club

The typical ML task in this dataset is to build a model that determines whether a loan will be approved.

For further details regarding the dataset features see
https://figshare.com/articles/dataset/Lending_club_dataset_description/20016077
"""
import typing as t
import warnings
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import sklearn
from category_encoders import OrdinalEncoder
from sklearn.compose import ColumnTransformer

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa # pylint: disable=unused-import

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data', 'load_fitted_model']

from deepchecks.utils.function import run_available_kwargs

_MODEL_URL = 'https://ndownloader.figshare.com/files/35692190'
_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/35685218'
_TRAIN_DATA_URL = 'https://ndownloader.figshare.com/files/35684222'
_TEST_DATA_URL = 'https://ndownloader.figshare.com/files/35684816'
_MODEL_VERSION = '1.0.2'
_target = 'loan_status'
_datetime_name = 'issue_d'
_index_name = 'id'
_CAT_FEATURES = ['addr_state', 'application_type', 'home_ownership', 'initial_list_status', 'purpose', 'term',
                 'verification_status', 'sub_grade']
_NUM_FEATURES = ['fico_range_low', 'total_acc', 'pub_rec', 'revol_util', 'annual_inc', 'int_rate', 'dti', 'mort_acc',
                 'loan_amnt', 'installment', 'pub_rec_bankruptcies', 'fico_range_high', 'revol_bal', 'open_acc',
                 'emp_length', 'time_to_earliest_cr_line']
_FEATURES = _CAT_FEATURES + _NUM_FEATURES


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> t.Union[
        t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns part of the Lending club dataset (classification).

    Parameters
    ----------
    data_format : str, default: 'Dataset'
        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
        'Dataset' will return the data as a Dataset object
        'Dataframe' will return the data as a pandas Dataframe object

    as_train_test : bool, default: True
        If True, the returned data is splitted into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.

    Returns
    -------
    dataset : Union[deepchecks.Dataset, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.
    """
    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL, index_col=False)

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES, index_name=_index_name,
                              datetime_name=_datetime_name)
            return dataset
        elif data_format == 'Dataframe':
            return dataset
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')
    else:
        train = pd.read_csv(_TRAIN_DATA_URL, index_col=False)
        test = pd.read_csv(_TEST_DATA_URL, index_col=False)

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES, index_name=_index_name,
                            datetime_name=_datetime_name)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES, index_name=_index_name,
                           datetime_name=_datetime_name)
            return train, test
        elif data_format == 'Dataframe':
            return train, test
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')


def load_fitted_model(pretrained=True):
    """Load and return a fitted classification model.

    Returns
    -------
    model : Joblib
        The model/pipeline that was trained on the adult dataset.

    """
    if sklearn.__version__ == _MODEL_VERSION and pretrained:
        with urlopen(_MODEL_URL) as f:
            model = joblib.load(f)
    else:
        model = _build_model()
        train, _ = load_data()
        model.fit(train.features_columns, train.data[_target])
    return model


def _build_model():
    """Build the model to fit."""
    categorical_transformer = Pipeline(steps=[('encoder',
                                               run_available_kwargs(OrdinalEncoder, handle_unknown='use_encoded_value',
                                                                    unknown_value=np.nan, dtype=np.float64))])
    preprocessor = ColumnTransformer(
        transformers=[('num', 'passthrough', _NUM_FEATURES), ('cat', categorical_transformer, _CAT_FEATURES), ])

    model = Pipeline(steps=[('preprocessing', preprocessor), ('model',
                                                              run_available_kwargs(HistGradientBoostingClassifier,
                                                                                   max_depth=5, max_iter=200,
                                                                                   random_state=42,
                                                                                   categorical_features=[False] * len(
                                                                                       _NUM_FEATURES) + [True] * len(
                                                                                       _CAT_FEATURES)))])
    return model
