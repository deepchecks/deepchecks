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
import sys

import numpy as np
import sklearn
from deepdiff import DeepDiff
from hamcrest import assert_that, instance_of
from sklearn.base import BaseEstimator

from deepchecks.tabular.datasets.classification import adult, breast_cancer, iris, lending_club, phishing
from deepchecks.tabular.datasets.regression import avocado, wine_quality


def assert_sklearn_model_params_equals(model1, model2):
    assert type(model1) == type(model2)
    diff = DeepDiff(model1.get_params(), model2.get_params(), ignore_numeric_type_changes=True)
    assert diff == {}


def assert_model_predict_on_data(model, train, test):
    assert_that(model.predict(train.features_columns.iloc[:1]), instance_of(np.ndarray))
    assert_that(model.predict(test.features_columns.iloc[:1]), instance_of(np.ndarray))


def assert_dataset_module(dataset_module):
    train, test = dataset_module.load_data()
    trained_model = dataset_module.load_fitted_model(pretrained=False)
    assert_model_predict_on_data(trained_model, train, test)

    # The models were trained on python 3.8, therefore tests for equality of pretrained only on this version
    python_minor_version = sys.version_info[1]
    if python_minor_version == 8:
        if hasattr(dataset_module, '_MODEL_VERSION'):
            if sklearn.__version__ != dataset_module._MODEL_VERSION:
                raise Exception(f'Can\'t test pretrained model for non matching sklearn version {sklearn.__version__}')
        pretrained_model = dataset_module.load_fitted_model(pretrained=True)
        if isinstance(pretrained_model, BaseEstimator):
            assert_sklearn_model_params_equals(pretrained_model, trained_model)
        assert_model_predict_on_data(pretrained_model, train, test)


def test_model_predict_on_breast_cancer():
    assert_dataset_module(breast_cancer)


def test_model_predict_on_iris():
    assert_dataset_module(iris)


def test_model_predict_on_phishing():
    assert_dataset_module(phishing)


def test_model_predict_on_adult():
    assert_dataset_module(adult)


def test_model_predict_on_avocado():
    assert_dataset_module(avocado)


def test_model_predict_on_lending_club():
    assert_dataset_module(lending_club)


def test_model_predict_on_wine_quality():
    assert_dataset_module(wine_quality)
