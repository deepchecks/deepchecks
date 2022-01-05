# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The phishing dataset contains a slightly synthetic dataset of urls - some regular and some used for phishing."""
import typing as t
import pandas as pd
import joblib
from urllib.request import urlopen
from deepchecks import Dataset

__all__ = ['load_data', 'load_fitted_model']

_MODEL_URL = 'https://figshare.com/ndownloader/files/32594447'
_FULL_DATA_URL = 'https://figshare.com/ndownloader/files/32553581'
_TRAIN_DATA_URL = 'https://figshare.com/ndownloader/files/32593298'
_TEST_DATA_URL = 'https://figshare.com/ndownloader/files/32593373'
_target = 'target'
_CAT_FEATURES = ['ext']


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the phishing url dataset (classification).

    The phishing url dataset contains slighly synthetic dataset of urls - some regular and some used for phishing.

    The dataset is based on the `great project <https://github.com/Rohith-2/url_classification_dl>`_ by
    `Rohith Ramakrishnan <https://www.linkedin.com/in/rohith-ramakrishnan-54094a1a0/>`_ and others, accompanied by
    a `blog post <https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d>`_.
    The authors have released it under an open license per our request, and for that we are very grateful to them.

    This dataset is licensed under the `Creative Commons Zero v1.0 Universal (CC0 1.0)
    <https://creativecommons.org/publicdomain/zero/1.0/>`_.

    The typical ML task in this dataset is to build a model that predicts the if the url is part of a phishing attack.

    Dataset Shape:
        .. list-table:: Dataset Shape
           :widths: 50 50
           :header-rows: 1

           * - Property
             - Value
           * - Samples Total
             - 11.35K
           * - Dimensionality
             - 25
           * - Features
             - real, string
           * - Targets
             - boolean

    Description:
        .. list-table:: Dataset Description
           :widths: 50 50 50
           :header-rows: 1

           * - Column name
             - Column Role
             - Description
           * - target
             - Label
             - 0 if the URL is benign, 1 if it is related to phishing
           * - month
             - Data
             - The month this URL was first encountered, as an int
           * - scrape_date
             - Date
             - The exact date this URL was first encountered
           * - ext
             - Feature
             - The domain extension
           * - urlLength
             - Feature
             - The number of characters in the URL
           * - numDigits
             - Feature
             - The number of digits in the URL
           * - numParams
             - Feature
             - The number of query parameters in the URL
           * - num_%20
             - Feature
             - The number of '%20' substrings in the URL
           * - num_@
             - Feature
             - The number of @ characters in the URL
           * - entropy
             - Feature
             - The entropy of the URL
           * - has_ip
             - Feature
             - True if the URL string contains an IP address
           * - hasHttp
             - Feature
             - True if the url's domain supports http
           * - hasHttps
             - Feature
             - True if the url's domain supports https
           * - urlIsLive
             - Feature
             - The URL was live at the time of scraping
           * - dsr
             - Feature
             - The number of days since domain registration
           * - dse
             - Feature
             - The number of days since domain registration expired
           * - bodyLength
             - Feature
             - The number of characters in the URL's web page
           * - numTitles
             - Feature
             - The number of HTML titles (H1/H2/...) in the page
           * - numImages
             - Feature
             - The number of images in the page
           * - numLinks
             - Feature
             - The number of links in the page
           * - specialChars
             - Feature
             - The number of special characters in the page
           * - scriptLength
             - Feature
             - The number of characters in scripts embedded in the page
           * - sbr
             - Feature
             - The ratio of scriptLength to bodyLength (`= scriptLength / bodyLength`)
           * - bscr
             - Feature
             - The ratio of bodyLength to specialChars (`= specialChars / bodyLength`)
           * - sscr
             - Feature
             - The ratio of scriptLength to specialChars (`= scriptLength / specialChars`)

    Args:
        data_format (str, default 'Dataset'):
            Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
            'Dataset' will return the data as a Dataset object
            'Dataframe' will return the data as a pandas Dataframe object

        as_train_test (bool, default False):
            If True, the returned data is splitted into train and test exactly like the toy model
            was trained. The first return value is the train data and the second is the test data.
            In order to get this model, call the load_fitted_model() function.
            Otherwise, returns a single object.

    Returns:
        data (Union[deepchecks.Dataset, pd.DataFrame]): the data object, corresponding to the data_format attribute.

        (train_data, test_data) (Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]):
           tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.
    """
    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL, index_col=0)

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label='target', cat_features=_CAT_FEATURES, datetime_name='scrape_date')

        return dataset
    else:
        train = pd.read_csv(_TRAIN_DATA_URL, index_col=0)
        test = pd.read_csv(_TEST_DATA_URL, index_col=0)

        if data_format == 'Dataset':
            train = Dataset(train, label='target', cat_features=_CAT_FEATURES, datetime_name='scrape_date')
            test = Dataset(test, label='target', cat_features=_CAT_FEATURES, datetime_name='scrape_date')

        return train, test


def load_fitted_model():
    """Load and return a fitted regression model to predict the target in the phishing dataset.

    Returns:
        model (Joblib model) the model/pipeline that was trained on the phishing dataset.

    """
    with urlopen(_MODEL_URL) as f:
        model = joblib.load(f)

    return model
