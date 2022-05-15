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
import sys

import numpy as np
import sklearn
from hamcrest import assert_that, instance_of

from deepchecks.tabular.datasets.classification import breast_cancer, iris, phishing, adult
from deepchecks.tabular.datasets.regression import avocado
from deepchecks.utils.model import get_model_of_pipeline


def assert_sklearn_model_equals(model1, model2):
    model1 = get_model_of_pipeline(model1)
    model2 = get_model_of_pipeline(model2)
    assert type(model1) == type(model2)
    if hasattr(model1, 'tree_'):
        assert (model1.tree_.value == model2.tree_.value).all()
    elif hasattr(model1, 'estimators_'):
        assert len(model1.estimators_) == len(model2.estimators_)
        for sub1, sub2 in zip(model1.estimators_, model2.estimators_):
            assert_sklearn_model_equals(sub1, sub2)
    else:
        raise Exception('Don\'t know how to compare models')


def assert_dataset_module(dataset_module):
    train, test = dataset_module.load_data()
    trained_model = dataset_module.load_fitted_model(pretrained=False)

    assert_that(trained_model.predict(train.features_columns.iloc[:1]), instance_of(np.ndarray))
    assert_that(trained_model.predict(test.features_columns.iloc[:1]), instance_of(np.ndarray))

    # The models were trained on python 3.8, therefore tests for equality of pretrained only on this version
    python_minor_version = sys.version_info[1]
    if python_minor_version == 8:
        if sklearn.__version__ != dataset_module._MODEL_VERSION:
            raise Exception(f'Can\'t test pretrained model for non matching sklearn version {sklearn.__version__}')
        pretrained_model = dataset_module.load_fitted_model(pretrained=True)
        assert_sklearn_model_equals(pretrained_model, trained_model)


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
