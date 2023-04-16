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
"""Dataset containing tweet and metadata information for multiclass prediction for emotion of tweet.

The data has 4653 tweet records with 5 features and a multiclass target column, referring to the emotion of the tweet.

This dataset is an extension of Cardiff's tweet_eval dataset,
For additional details about the dataset, please refer to the original source: https://github.com/cardiffnlp/tweeteval.
Dataset originally published in "Semeval-2018 task 1: Affect in tweets" by Mohammad et al. (2018):
https://aclanthology.org/S18-1001/.
"""
import os
import pathlib
import typing as t

import numpy as np
import pandas as pd

from deepchecks.nlp import TextData

__all__ = ['load_data', 'load_embeddings', 'load_precalculated_predictions']

_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/39486889'
_EMBEDDINGS_URL = 'https://ndownloader.figshare.com/files/39264332'
_PROPERTIES_URL = 'https://ndownloader.figshare.com/files/39717619'
_PREDICTIONS_URL = 'https://ndownloader.figshare.com/files/39264461'

ASSETS_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'tweet_emotion'
_target = 'label'

_LABEL_MAP = {0: 'anger', 1: 'happiness', 2: 'optimism', 3: 'sadness'}
_CAT_METADATA = ['gender', 'user_region']
_CAT_PROPERTIES = ['Language']


def load_properties(as_train_test: bool = True) -> t.Union[pd.DataFrame, t.Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load and return the properties of the tweet_emotion dataset.

    Parameters
    ----------
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
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
        return properties.drop(columns=['train_test_split']).sort_index()


def load_data(data_format: str = 'TextData', as_train_test: bool = True,
              include_properties: bool = True) -> \
        t.Union[t.Tuple, t.Union[TextData, pd.DataFrame]]:
    """Load and returns the Tweet Emotion dataset (classification).

    Parameters
    ----------
    data_format : str, default: 'TextData'
        Represent the format of the returned value. Can be 'TextData'|'DataFrame'
        'TextData' will return the data as a TextData object
        'Dataframe' will return the data as a pandas DataFrame object
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
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
        tuple if as_train_test = True. Tuple of two objects represents the dataset split to train and test sets.
    """
    if data_format.lower() not in ['textdata', 'dataframe']:
        raise ValueError('data_format must be either "Dataset" or "Dataframe"')

    data = _read_and_save('tweet_emotion_data.csv', _FULL_DATA_URL, to_numpy=False)
    if not as_train_test:
        data.drop(columns=['train_test_split'], inplace=True)
        if data_format.lower() != 'textdata':
            return data

        dataset = TextData(data.text, label=data[_target], task_type='text_classification')
        dataset.set_metadata(metadata=data.drop(columns=[_target, 'text']),
                             categorical_metadata=_CAT_METADATA)
        if include_properties:
            properties = load_properties(as_train_test=False)
            dataset.set_properties(properties=properties, categorical_properties=_CAT_PROPERTIES)
        return dataset
    else:
        # train has more sport and Customer Complains but less Terror and Optimism
        train = data[data['train_test_split'] == 'Train'].drop(columns=['train_test_split'])
        test = data[data['train_test_split'] == 'Test'].drop(columns=['train_test_split'])

        if data_format.lower() != 'textdata':
            return train, test

        train_ds = TextData(train.text, label=train[_target], task_type='text_classification')
        train_ds.set_metadata(metadata=train.drop(columns=[_target, 'text']),
                           categorical_metadata=_CAT_METADATA)
        test_ds = TextData(test.text, label=test[_target], task_type='text_classification')
        test_ds.set_metadata(metadata=test.drop(columns=[_target, 'text']),
                            categorical_metadata=_CAT_METADATA)

        if include_properties:
            train_properties, test_properties = load_properties(as_train_test=True)
            train_ds.set_properties(properties=train_properties, categorical_properties=_CAT_PROPERTIES)
            test_ds.set_properties(properties=test_properties, categorical_properties=_CAT_PROPERTIES)

        return train_ds, test_ds


def load_embeddings(as_train_test: bool = False) -> np.ndarray:
    """Load and return the embeddings of the tweet_emotion dataset calculated by OpenAI.

    Parameters
    ----------
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        Otherwise, returns a single object.

    Returns
    -------
    embeddings : np.ndarray
        Embeddings for the tweet_emotion dataset.
    """
    all_embeddings = _read_and_save('tweet_emotion_embeddings.csv', _EMBEDDINGS_URL)
    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes()
        return all_embeddings[train_indexes], all_embeddings[test_indexes]
    else:
        return all_embeddings


def load_precalculated_predictions(pred_format: str = 'predictions',
                                   as_train_test: bool = True) -> np.array:
    """Load and return a precalculated predictions for the dataset.

    Parameters
    ----------
    pred_format : str, default: 'predictions'
        Represent the format of the returned value. Can be 'predictions' or 'probabilities'.
        'predictions' will return the predicted class for each sample.
        'probabilities' will return the predicted probabilities for each sample.
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        Otherwise, returns a single object.

    Returns
    -------
    predictions : np.ndarray
        The prediction of the data elements in the dataset.

    """
    all_preds = _read_and_save('tweet_emotion_probabilities.csv', _PREDICTIONS_URL)
    if pred_format == 'predictions':
        all_preds = np.array([_LABEL_MAP[x] for x in np.argmax(all_preds, axis=1)])
    elif pred_format != 'probabilities':
        raise ValueError('pred_format must be either "predictions" or "probabilities"')

    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes()
        return all_preds[train_indexes], all_preds[test_indexes]
    else:
        return all_preds


def _read_and_save(file_name, url_to_file, to_numpy=True):
    """Read a file from a url and save it to the assets directory."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if (ASSETS_DIR / file_name).exists():
        data = pd.read_csv(ASSETS_DIR / file_name, index_col=0)
    else:
        data = pd.read_csv(url_to_file, index_col=0)
        data.to_csv(ASSETS_DIR / file_name)

    if to_numpy:
        data = data.to_numpy()
    return data


def _get_train_test_indexes() -> t.Tuple[np.array, np.array]:
    """Get the indexes of the train and test sets."""
    if (ASSETS_DIR / 'tweet_emotion_data.csv').exists():
        dataset = pd.read_csv(ASSETS_DIR / 'tweet_emotion_data.csv', index_col=0,
                              usecols=['Unnamed: 0', 'train_test_split'])
    else:
        dataset = pd.read_csv(_FULL_DATA_URL, index_col=0, usecols=['Unnamed: 0', 'train_test_split'])

    train_indexes = dataset[dataset['train_test_split'] == 'Train'].index
    test_indexes = dataset[dataset['train_test_split'] == 'Test'].index
    return train_indexes, test_indexes
