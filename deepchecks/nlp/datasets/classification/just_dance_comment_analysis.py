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
"""Dataset containing comments and metadata information for multilabel predictions for different properties of comments.

The data has 216193 comments make on the just dance YouTube videos. It has metadata information about the date the
comment was written and the number of "likes" it got. It also has
42 multilabel binary target label columns,
referring to the category classification of the comment.

This dataset is a modification of Just Dance @ YouTube dataset curated by the COIMBRA university,
For additional details about the dataset, please refer to the original source:
https://www.kaggle.com/datasets/renatojmsantos/just-dance-on-youtube.
Dataset used under the following license: https://creativecommons.org/licenses/by/4.0/

Original publication:
R. Santos, J. P. Arrais and P. A. Silva, "Analysing Games for Health through Users' Opinion Mining,"
2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS), Aveiro, Portugal, 2021, pp. 319-323,
doi: 10.1109/CBMS52027.2021.00035.
"""
import pathlib
import typing as t
import warnings

import numpy as np
import pandas as pd

from deepchecks.nlp import TextData
from deepchecks.utils.builtin_datasets_utils import read_and_save_data

__all__ = ['load_data']

_FULL_DATA_URL = 'https://figshare.com/ndownloader/files/40564895'
_SHORT_DATA_URL = 'https://figshare.com/ndownloader/files/40576232'
_SHORT_PROPERTIES_URL = 'https://figshare.com/ndownloader/files/40580693'
_SHORT_EMBEDDINGS_URL = 'https://figshare.com/ndownloader/files/40576328'
_SHORT_PROBAS_URL = 'https://figshare.com/ndownloader/files/40578866'

ASSETS_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'just_dance_comment_analysis'

_METADATA_COLS = ['likes', 'dateComment']
_CAT_METADATA = []
_TEXT_COL = 'originalText'
_TIME_COL = 'dateComment'
_DATE_TO_SPLIT_BY = '2015-01-01'


def load_precalculated_predictions(pred_format: str = 'predictions', as_train_test: bool = True,
                                   use_full_size: bool = False) -> \
        t.Union[np.array, t.Tuple[np.array, np.array]]:
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
    use_full_size : bool, default: False
        If True, the returned data will be the full dataset, otherwise returns a subset of the data.
    Returns
    -------
    predictions : np.ndarray
        The prediction of the data elements in the dataset.

    """
    if use_full_size:
        raise NotImplementedError('Predictions for the full dataset are not yet available.')
    all_preds = read_and_save_data(ASSETS_DIR, 'just_dance_probabilities.csv', _SHORT_PROBAS_URL, to_numpy=True,
                                   file_type='npy')

    if pred_format == 'predictions':
        all_preds = (np.array(all_preds) > 0.5)
        all_preds = all_preds.astype(int)
    elif pred_format != 'probabilities':
        raise ValueError('pred_format must be either "predictions" or "probabilities"')

    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes()
        return all_preds[train_indexes], all_preds[test_indexes]
    else:
        return all_preds


def load_embeddings(as_train_test: bool = True, use_full_size: bool = False) -> \
        t.Union[np.array, t.Tuple[np.array, np.array]]:
    """Load and return the embeddings of the just dance dataset calculated by OpenAI.

    Parameters
    ----------
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        Otherwise, returns a single object.
    use_full_size : bool, default: False
        If True, the returned data will be the full dataset, otherwise returns a subset of the data.

    Returns
    -------
    embeddings : np.ndarray
        Embeddings for the just dance dataset.
    """
    if use_full_size:
        raise NotImplementedError('Embeddings for the full dataset are not yet available.')

    all_embeddings = read_and_save_data(ASSETS_DIR, 'just_dance_embeddings.npy', _SHORT_EMBEDDINGS_URL,
                                        file_type='npy', to_numpy=True)

    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes(use_full_size)
        return all_embeddings[train_indexes], all_embeddings[test_indexes]
    else:
        return all_embeddings


def load_properties(as_train_test: bool = True, use_full_size: bool = False) -> \
        t.Union[pd.DataFrame, t.Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load and return the properties of the just_dance dataset.

    Parameters
    ----------
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.
    use_full_size : bool, default: False
        If True, the returned data will be the full dataset, otherwise returns a subset of the data.
    Returns
    -------
    properties : pd.DataFrame
        Properties for the just dance dataset.
    """
    if use_full_size:
        raise NotImplementedError('Properties for the full dataset are not yet available.')
    properties = read_and_save_data(ASSETS_DIR, 'just_dance_properties.csv', _SHORT_PROPERTIES_URL, to_numpy=False)

    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes(use_full_size)
        return properties.loc[train_indexes], properties.loc[test_indexes]
    else:
        return properties


def load_data(data_format: str = 'TextData', as_train_test: bool = True, use_full_size: bool = False,
              include_properties: bool = True, include_embeddings: bool = False) -> \
        t.Union[t.Tuple, t.Union[TextData, pd.DataFrame]]:
    """Load and returns the Just Dance Comment Analysis dataset (multi-label classification).

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
    use_full_size : bool, default: False
        If True, the returned data will be the full dataset, otherwise returns a subset of the data.
    include_properties : bool, default: True
        If True, the returned data will include properties of the comments. Incompatible with data_format='DataFrame'
    include_embeddings : bool, default: False
        If True, the returned data will include embeddings of the comments. Incompatible with data_format='DataFrame'

    Returns
    -------
    dataset : Union[TextData, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[TextData, pd.DataFrame],Union[TextData, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset split to train and test sets.
    """
    if data_format.lower() not in ['textdata', 'dataframe']:
        raise ValueError('data_format must be either "TextData" or "Dataframe"')
    elif data_format.lower() == 'dataframe':
        if include_properties or include_embeddings:
            warnings.warn('include_properties and include_embeddings are incompatible with data_format="Dataframe". '
                          'loading only original text data.',
                          UserWarning)
            include_properties, include_embeddings = False, False

    if use_full_size:
        data = read_and_save_data(ASSETS_DIR, 'just_dance_data.csv', _FULL_DATA_URL, to_numpy=False,
                                  include_index=False)
    else:
        data = read_and_save_data(ASSETS_DIR, 'just_dance_shorted_data.csv', _SHORT_DATA_URL, to_numpy=False)
    data[_TIME_COL] = pd.to_datetime(data[_TIME_COL])

    properties = load_properties(as_train_test=False, use_full_size=use_full_size) if include_properties else None
    embeddings = load_embeddings(as_train_test=False, use_full_size=use_full_size) if include_embeddings else None

    if not as_train_test:
        if data_format.lower() != 'textdata':
            return data

        label = data.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)
        dataset = TextData(data[_TEXT_COL], label=label, task_type='text_classification',
                           metadata=data[_METADATA_COLS], categorical_metadata=_CAT_METADATA,
                           properties=properties, embeddings=embeddings)
        return dataset

    else:
        train_indexes, test_indexes = _get_train_test_indexes(use_full_size)
        train, test = data.loc[train_indexes], data.loc[test_indexes]

        if data_format.lower() != 'textdata':
            return train, test

        train_metadata, test_metadata = train[_METADATA_COLS], test[_METADATA_COLS]
        label_train = train.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)
        label_test = test.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)

        if include_properties:
            train_properties, test_properties = properties.loc[train.index], properties.loc[test.index]
        else:
            train_properties, test_properties = None, None
        if include_embeddings:
            train_embeddings = embeddings[train.index]  # pylint: disable=unsubscriptable-object
            test_embeddings = embeddings[test.index]  # pylint: disable=unsubscriptable-object
        else:
            train_embeddings, test_embeddings = None, None

        train_ds = TextData(train[_TEXT_COL], label=label_train, task_type='text_classification',
                            metadata=train_metadata, categorical_metadata=_CAT_METADATA,
                            properties=train_properties, embeddings=train_embeddings)
        test_ds = TextData(test[_TEXT_COL], label=label_test, task_type='text_classification',
                           metadata=test_metadata, categorical_metadata=_CAT_METADATA,
                           properties=test_properties, embeddings=test_embeddings)

        return train_ds, test_ds


def _get_train_test_indexes(use_full_size: bool = False) -> t.Tuple[np.array, np.array]:
    """Get the indexes of the train and test sets."""
    if use_full_size:
        dataset = pd.read_csv(ASSETS_DIR / 'just_dance_data.csv', usecols=[_TIME_COL])
    else:
        dataset = pd.read_csv(ASSETS_DIR / 'just_dance_shorted_data.csv', usecols=[_TIME_COL])
    train_indexes = dataset[dataset[_TIME_COL] < _DATE_TO_SPLIT_BY].index
    test_indexes = dataset[dataset[_TIME_COL] >= _DATE_TO_SPLIT_BY].index
    return train_indexes, test_indexes
