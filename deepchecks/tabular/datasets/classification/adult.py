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
"""The data set contains features for binary prediction of the income of an adult (the adult dataset).

The data has 48842 records with 14 features and one binary target column, referring to whether the person's income
is greater than 50K.

This is a copy of UCI ML Adult dataset. https://archive.ics.uci.edu/ml/datasets/adult

References:
    * Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",
      Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

The typical ML task in this dataset is to build a model that determines whether a person makes over 50K a year.

Dataset Shape:
    .. list-table:: Dataset Shape
       :widths: 50 50
        :header-rows: 1

       * - Property
         - Value
        * - Samples Total
         - 48842
        * - Dimensionality
         - 14
        * - Features
         - real, string
        * - Targets
         - 2
        * - Samples per class
         - '>50K' - 23.93%, '<=50K' - 76.07%

Description:
    .. list-table:: Dataset Description
       :widths: 50 50 50
       :header-rows: 1

       * - Column name
         - Column Role
         - Description
       * - Age
         - Feature
         - The age of the person.
       * - workclass
         - Feature
         - [Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked]
       * - fnlwgt
         - Feature
         - Final weight.
       * - education
         - Feature
         - [Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters,
            1st-4th, 10th, Doctorate, 5th-6th, Preschool]
       * - education-num
         - Feature
         - Number of years of education
       * - marital-status
         - Feature
         - [Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,
            Married-AF-spouse]
       * - occupation
         - Feature
         - [Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
            Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
            Armed-Forces]
       * - relationship
         - Feature
         - [Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried]
       * - race
         - Feature
         - [White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black]
       * - sex
         - Feature
         - [Male, Female]
       * - capital-gain
         - Feature
         - The capital gain of the person
       * - capital-loss
         - Feature
         - The capital loss of the person
       * - hours-per-week
         - Feature
         - The number of hours worked per week
       * - native-country
         - Feature
         - [United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India,
            Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
            Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary,
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong,
            Holand-Netherlands]
       * - target
         - Target
         - The target variable, whether the person makes over 50K a year.
"""
import typing as t
from urllib.request import urlopen

import joblib
import pandas as pd
import sklearn
from category_encoders import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from deepchecks.tabular.dataset import Dataset

__all__ = ['load_data', 'load_fitted_model']

_MODEL_URL = 'https://figshare.com/ndownloader/files/35122753'
_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/34516457'
_TRAIN_DATA_URL = 'https://ndownloader.figshare.com/files/34516448'
_TEST_DATA_URL = 'https://ndownloader.figshare.com/files/34516454'
_MODEL_VERSION = '1.0.2'
_FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
_target = 'income'
_CAT_FEATURES = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'native-country']
_NUM_FEATURES = sorted(list(set(_FEATURES) - set(_CAT_FEATURES)))


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Adult dataset (classification).

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
        dataset = pd.read_csv(_FULL_DATA_URL, names=_FEATURES + [_target])

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES)
            return dataset
        elif data_format == 'Dataframe':
            return dataset
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')
    else:
        train = pd.read_csv(_TRAIN_DATA_URL, names=_FEATURES + [_target])
        test = pd.read_csv(_TEST_DATA_URL, skiprows=1, names=_FEATURES + [_target])
        test[_target] = test[_target].str[:-1]

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES)
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
        model.fit(train.data[train.features], train.data[train.label_name])
    return model


def _build_model():
    """Build the model to fit."""
    numeric_transformer = SimpleImputer()
    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, _NUM_FEATURES),
            ('cat', categorical_transformer, _CAT_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('model', RandomForestClassifier(max_depth=5, n_jobs=-1, random_state=0))
        ]
    )

    return model
