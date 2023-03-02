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
"""Dataset containing tweets and meta data information for multiclass prediction for emotion of tweet.

The data has 4653 tweet records with 5 features and a multiclass target column, referring to the emotion of the tweet.

This dataset is an extension of Cardiff's tweet_eval dataset,
For additional details about the dataset, please refer to the original source: https://github.com/cardiffnlp/tweeteval
"""
import os
import pathlib
import typing as t

import numpy as np
import pandas as pd

from deepchecks.nlp import TextData

__all__ = ['load_data', 'load_embeddings', 'load_precalculated_predictions']

_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/39265559'
_EMBEDDINGS_URL = 'https://ndownloader.figshare.com/files/39264332'
_PROPERTIES_URL = 'https://ndownloader.figshare.com/files/39460924'
_PREDICTIONS_URL = 'https://ndownloader.figshare.com/files/39264461'

ASSETS_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'tweet_emotion'
_target = 'label'


def load_properties(as_train_test: bool = True) -> t.Union[pd.DataFrame, t.Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load and return the properties of the tweet_emotion dataset.

    Parameters
    ----------
    as_train_test : bool, default: True
        If True, the returned data is splitted into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.

    Returns
    -------
    properties : pd.DataFrame
        Properties for the tweet_emotion dataset.
    """
    if (ASSETS_DIR / 'tweet_emotion_properties.csv').exists():
        properties = pd.read_csv(ASSETS_DIR / 'tweet_emotion_properties.csv', index_col=0)
    else:
        properties = pd.read_csv(_PROPERTIES_URL, index_col=0)
        properties.to_csv(ASSETS_DIR / 'tweet_emotion_properties.csv')

    if as_train_test:
        train = properties[properties['train_test_split'] == 'Train'].drop(columns=['train_test_split'])
        test = properties[properties['train_test_split'] == 'Test'].drop(columns=['train_test_split'])
        return train, test
    else:
        return properties.drop(columns=['train_test_split'])


def load_data(data_format: str = 'TextData', as_train_test: bool = True,
              include_properties: bool = True) -> \
        t.Union[t.Tuple, t.Union[TextData, pd.DataFrame]]:
    """Load and returns the Breast Cancer dataset (classification).

    Parameters
    ----------
    data_format : str, default: 'TextData'
        Represent the format of the returned value. Can be 'TextData'|'DataFrame'
        'TextData' will return the data as a TextData object
        'Dataframe' will return the data as a pandas DataFrame object
    as_train_test : bool, default: True
        If True, the returned data is splitted into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.
    include_properties : bool, default: True
        If True, the returned data will include the properties of the tweets. Incompatible with data_format='DataFrame'

    Returns
    -------
    dataset : Union[TextData, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[TextData, pd.DataFrame],Union[TextData, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.
    """
    if data_format.lower() not in ['textdata', 'dataframe']:
        raise ValueError('data_format must be either "Dataset" or "Dataframe"')

    os.makedirs(ASSETS_DIR, exist_ok=True)
    if (ASSETS_DIR / 'tweet_emotion_data.csv').exists():
        dataset = pd.read_csv(ASSETS_DIR / 'tweet_emotion_data.csv', index_col=0)
    else:
        dataset = pd.read_csv(_FULL_DATA_URL, index_col=0)
        dataset.to_csv(ASSETS_DIR / 'tweet_emotion_data.csv')

    if not as_train_test:
        dataset.drop(columns=['train_test_split'], inplace=True)
        if data_format.lower() == 'textdata':
            if include_properties:
                properties = load_properties(as_train_test=False)
            else:
                properties = None
            dataset = TextData(dataset.text, label=dataset[_target], task_type='text_classification',
                               additional_data=dataset.drop(columns=[_target, 'text']),
                               properties=properties, index=dataset.index)
        return dataset
    else:
        # train has more sport and Customer Complains but less Terror and Optimism
        train = dataset[dataset['train_test_split'] == 'Train'].drop(columns=['train_test_split'])
        test = dataset[dataset['train_test_split'] == 'Test'].drop(columns=['train_test_split'])

        if data_format.lower() == 'textdata':
            if include_properties:
                train_properties, test_properties = load_properties(as_train_test=True)
            else:
                train_properties, test_properties = None, None

            train = TextData(train.text, label=train[_target], task_type='text_classification',
                             index=train.index, additional_data=train.drop(columns=[_target, 'text']),
                             properties=train_properties)
            test = TextData(test.text, label=test[_target], task_type='text_classification',
                            index=test.index, additional_data=test.drop(columns=[_target, 'text']),
                            properties=test_properties)
        return train, test


def load_embeddings() -> np.ndarray:
    """Load and return the embeddings of the tweet_emotion dataset calculated by OpenAI.

    Returns
    -------
    embeddings : np.ndarray
        Embeddings for the tweet_emotion dataset.
    """
    return pd.read_csv(_EMBEDDINGS_URL, index_col=0).to_numpy()


def load_precalculated_predictions(pred_format: str = 'predictions') -> np.ndarray:
    """Load and return a precalculated predictions for the dataset.

    Parameters
    ----------
    pred_format : str, default: 'predictions'
        Represent the format of the returned value. Can be 'predictions' or 'probabilities'.
        'predictions' will return the predicted class for each sample.
        'probabilities' will return the predicted probabilities for each sample.

    Returns
    -------
    predictions : np.ndarray
        The prediction of the data elements in the dataset.

    """
    preds = pd.read_csv(_PREDICTIONS_URL, index_col=0).to_numpy()
    if pred_format == 'predictions':
        return np.argmax(preds, axis=1)
    elif pred_format == 'probabilities':
        return preds
    else:
        raise ValueError('pred_format must be either "predictions" or "probabilities"')
