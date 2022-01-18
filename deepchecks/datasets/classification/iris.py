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
"""The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant."""
import typing as t
import pandas as pd
import joblib
from urllib.request import urlopen
from deepchecks import Dataset

__all__ = ['load_data', 'load_fitted_model']

_MODEL_URL = 'https://figshare.com/ndownloader/files/32653100'
_FULL_DATA_URL = 'https://figshare.com/ndownloader/files/32652977'
_TRAIN_DATA_URL = 'https://figshare.com/ndownloader/files/32653172'
_TEST_DATA_URL = 'https://figshare.com/ndownloader/files/32653130'
_target = 'target'
_CAT_FEATURES = []


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Iris dataset (classification).

    The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
    One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

    References:
        * Fisher, R.A. “The use of multiple measurements in taxonomic problems” Annual Eugenics, 7, Part II,
          179-188 (1936); also in “Contributions to Mathematical Statistics” (John Wiley, NY, 1950).
        * Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis. (Q327.D83) John Wiley & Sons.
          ISBN 0-471-22361-1. See page 218.
        * And many more..

    The typical ML task in this dataset is to build a model that classifies the type of flower.

    Dataset Shape:
        .. list-table:: Dataset Shape
           :widths: 50 50
           :header-rows: 1

           * - Property
             - Value
           * - Samples Total
             - 150
           * - Dimensionality
             - 4
           * - Features
             - real
           * - Targets
             - 3
           * - Samples per class
             - 50

    Description:
        .. list-table:: Dataset Description
           :widths: 50 50 50
           :header-rows: 1

           * - Column name
             - Column Role
             - Description
           * - sepal length (cm)
             - Feature
             - The length of the flower's sepal (in cm)
           * - sepal width (cm)
             - Feature
             - The width of the flower's sepal (in cm)
           * - petal length (cm)
             - Feature
             - The length of the flower's petal (in cm)
           * - petal width (cm)
             - Feature
             - The width of the flower's petal (in cm)
           * - target
             - Label
             - The class (Setosa,Versicolour,Virginica)

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
        dataset = pd.read_csv(_FULL_DATA_URL)

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES)

        return dataset
    else:
        train = pd.read_csv(_TRAIN_DATA_URL)
        test = pd.read_csv(_TEST_DATA_URL)

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES)

        return train, test


def load_fitted_model():
    """Load and return a fitted classification model to predict the flower type in the iris dataset.

    Returns:
        model (Joblib model) the model/pipeline that was trained on the iris dataset.

    """
    with urlopen(_MODEL_URL) as f:
        model = joblib.load(f)

    return model
