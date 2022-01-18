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
"""The data set contains features for binary prediction of breast cancer."""
import typing as t
from urllib.request import urlopen

import joblib
import pandas as pd

from deepchecks import Dataset

__all__ = ['load_data', 'load_fitted_model']

_MODEL_URL = 'https://ndownloader.figshare.com/files/33325673'
_FULL_DATA_URL = 'https://ndownloader.figshare.com/files/33325472'
_TRAIN_DATA_URL = 'https://ndownloader.figshare.com/files/33325556'
_TEST_DATA_URL = 'https://ndownloader.figshare.com/files/33325559'
_target = 'target'
_CAT_FEATURES = []


def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Breast Cancer dataset (classification).

    The data has 569 patient records with 30 features and one binary target column, referring to the presence of
    breast cancer in the patient.

    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets. https://goo.gl/U2Uwz2

    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe
    characteristics of the cell nuclei present in the image.

    Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett,
    “Decision Tree Construction Via Linear Programming.” Proceedings of the 4th Midwest Artificial Intelligence and
    Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct
    a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3
    separating planes.

    The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [
    K. P. Bennett and O. L. Mangasarian: “Robust Linear Programming Discrimination of Two Linearly Inseparable Sets”,
    Optimization Methods and Software 1, 1992, 23-34].

    This database is also available through the UW CS ftp server:

    ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

    References:
        * W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor
          diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905,
          pages 861-870, San Jose, CA, 1993.
        * O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and
          prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.
        * W.H. Wolberg,
          W.N. Street, and O.L. Mangasarian. Machine learning techniques to diagnose breast cancer from fine-needle
          aspirates. Cancer Letters 77 (1994) 163-171.

    The typical ML task in this dataset is to build a model that classifies between benign and malignant samples.

    Ten real-valued features are computed for each cell nucleus:
        #. radius (mean of distances from center to points on the perimeter)
        #. texture (standard deviation of gray-scale values)
        #. perimeter
        #. area
        #. smoothness (local variation in radius lengths)
        #. compactness (perimeter^2 / area - 1.0)
        #. concavity (severity of concave portions of the contour)
        #. concave points (number of concave portions of the contour)
        #. symmetry
        #. fractal dimension ("coastline approximation" - 1)

    Dataset Shape:
        .. list-table:: Dataset Shape
           :widths: 50 50
           :header-rows: 1

           * - Property
             - Value
           * - Samples Total
             - 569
           * - Dimensionality
             - 30
           * - Features
             - real
           * - Targets
             - boolean


    Description:
        .. list-table:: Dataset Description
           :widths: 50 50 50
           :header-rows: 1

            * - mean radius
              - Feature
              - mean radius
            * - mean texture
              - Feature
              - mean texture
            * - mean perimeter
              - Feature
              - mean perimeter
            * - mean area
              - Feature
              - mean area
            * - mean smoothness
              - Feature
              - mean smoothness
            * - mean compactness
              - Feature
              - mean compactness
            * - mean concavity
              - Feature
              - mean concavity
            * - mean concave points
              - Feature
              - mean concave points
            * - mean symmetry
              - Feature
              - mean symmetry
            * - mean fractal dimension
              - Feature
              - mean fractal dimension
            * - radius error
              - Feature
              - radius error
            * - texture error
              - Feature
              - texture error
            * - perimeter error
              - Feature
              - perimeter error
            * - area error
              - Feature
              - area error
            * - smoothness error
              - Feature
              - smoothness error
            * - compactness error
              - Feature
              - compactness error
            * - concavity error
              - Feature
              - concavity error
            * - concave points error
              - Feature
              - concave points error
            * - symmetry error
              - Feature
              - symmetry error
            * - fractal dimension error
              - Feature
              - fractal dimension error
            * - worst radius
              - Feature
              - worst radius
            * - worst texture
              - Feature
              - worst texture
            * - worst perimeter
              - Feature
              - worst perimeter
            * - worst area
              - Feature
              - worst area
            * - worst smoothness
              - Feature
              - worst smoothness
            * - worst compactness
              - Feature
              - worst compactness
            * - worst concavity
              - Feature
              - worst concavity
            * - worst concave points
              - Feature
              - worst concave points
            * - worst symmetry
              - Feature
              - worst symmetry
            * - worst fractal dimension
              - Feature
              - worst fractal dimension
            * - target
              - Label
              - The class (Benign, Malignant)

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
        elif data_format == 'Dataframe':
            return dataset
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')
    else:
        train = pd.read_csv(_TRAIN_DATA_URL)
        test = pd.read_csv(_TEST_DATA_URL)

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES)
            return train, test
        elif data_format == 'Dataframe':
            return train, test
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')


def load_fitted_model():
    """Load and return a fitted classification model to predict the flower type in the iris dataset.

    Returns:
        model (Joblib model) the model/pipeline that was trained on the iris dataset.

    """
    with urlopen(_MODEL_URL) as f:
        model = joblib.load(f)

    return model
