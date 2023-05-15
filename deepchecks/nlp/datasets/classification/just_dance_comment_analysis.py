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

import pandas as pd

from deepchecks.nlp import TextData
from deepchecks.utils.builtin_datasets_utils import read_and_save_data

__all__ = ['load_data']


_FULL_DATA_URL = 'https://figshare.com/ndownloader/files/40564895'


ASSETS_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'just_dance_comment_analysis'

_METADATA_COLS = ['likes', 'dateComment']
_CAT_METADATA = []
_CAT_PROPERTIES = ['Language']
_TEXT_COL = 'originalText'


def load_data(data_format: str = 'TextData', as_train_test: bool = True, use_full_size: bool = False) -> \
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

    Returns
    -------
    dataset : Union[TextData, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[TextData, pd.DataFrame],Union[TextData, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset split to train and test sets.
    """
    if data_format.lower() not in ['textdata', 'dataframe']:
        raise ValueError('data_format must be either "Dataset" or "Dataframe"')

    data = read_and_save_data(ASSETS_DIR, 'just_dance_data.csv', _FULL_DATA_URL, to_numpy=False)
    data['dateComment'] = pd.to_datetime(data['dateComment'])

    if not as_train_test:
        if not use_full_size:
            data = data[(data['dateComment'] < '2013-01-01') | (data['dateComment'] >= '2021-01-01')]
        if data_format.lower() != 'textdata':
            return data

        label = data.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)
        dataset = TextData(data[_TEXT_COL], label=label, task_type='text_classification',
                           metadata=data[_METADATA_COLS], categorical_metadata=_CAT_METADATA)
        return dataset

    else:
        if use_full_size:
            train = data[data['dateComment'] < '2015-01-01']
            test = data[data['dateComment'] >= '2015-01-01']
        else:
            train = data[data['dateComment'] < '2013-01-01']
            test = data[data['dateComment'] >= '2021-01-01']

        if data_format.lower() != 'textdata':
            return train, test

        train_metadata, test_metadata = train[_METADATA_COLS], test[_METADATA_COLS]
        label_train = train.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)
        label_test = test.drop(columns=[_TEXT_COL] + _METADATA_COLS).to_numpy().astype(int)

        train_ds = TextData(train[_TEXT_COL], label=label_train, task_type='text_classification',
                            metadata=train_metadata, categorical_metadata=_CAT_METADATA)
        test_ds = TextData(test[_TEXT_COL], label=label_test, task_type='text_classification',
                           metadata=test_metadata, categorical_metadata=_CAT_METADATA)

        return train_ds, test_ds
