import typing as t
import pandas as pd
import joblib
from urllib.request import urlopen
from deepchecks import Dataset

__all__ = ['load_data', 'load_fitted_model']

_MODEL_URL = "https://figshare.com/ndownloader/files/32393723"
_FULL_DATA_URL = "https://figshare.com/ndownloader/files/32393729"
_TRAIN_DATA_URL = "https://figshare.com/ndownloader/files/32393732"
_TEST_DATA_URL = "https://figshare.com/ndownloader/files/32393726"
_target = "AveragePrice"
_CAT_FEATURES = ["region", "type"]


def load_data(format: str = 'Dataset', as_train_test: bool = True) -> t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Avocado dataset (regression).

    The avocado dataset contains historical data on avocado prices and sales volume in multiple US markets
    https://www.kaggle.com/neuromusic/avocado-prices.

    =================   ==================
    Samples total       18.2K
    Dimensionality      14
    Features            real, string
    Targets             real 0.44 - 3.25
    =================   ==================

    Dataset Description:

    ==============      =============
    Column name         Column Role
    Total Volume        Feature
    4046                Feature
    4225                Feature
    4770                Feature
    Total Bags          Feature
    Small Bags          Feature
    Large Bags          Feature
    XLarge Bags         Feature
    type                Feature
    year                Feature
    region              Feature
    AveragePrice        Label
    ==============      =============

    Args:
        format (str, default 'Dataset'):
            Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
            'Dataset' will return the data as a Dataset object
            'Dataframe' will return the data as a pandas Dataframe object

        as_train_test (bool, default False):
            If True, the returned data is splitted into train and test exactly like the toy model
            was trained. The first return value is the train data and the second is the test data.
            In order to get this model, call the load_fitted_model() function.
            Otherwise, returns a single object.

    Returns:
        data (Union[deepchecks.Dataset, pd.DataFrame]) the data object, corresponding to the format attribute.

        (train_data, test_data) (Tuple[Union[deepchecks.Dataset, pd.DataFrame], Union[deepchecks.Dataset, pd.DataFrame])
           tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.

    """

    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL)

        if format == 'Dataset':
            dataset = Dataset(dataset, label_name='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')

        return dataset
    else:
        train = pd.read_csv(_TRAIN_DATA_URL)
        test = pd.read_csv(_TEST_DATA_URL)

        if format == 'Dataset':
            train = Dataset(train, label_name='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')
            test = Dataset(test, label_name='AveragePrice', cat_features=_CAT_FEATURES, datetime_name='Date')

        return train, test


def load_fitted_model():
    """Load and return a fitted regression model to predict the AveragePrice in the avocado dataset.

    Returns:
        model (Joblib model) the model/pipeline that was trained on the Avocado dataset.

    """
    model = joblib.load(urlopen(_MODEL_URL))

    return model
