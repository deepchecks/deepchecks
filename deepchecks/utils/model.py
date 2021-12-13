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
"""Model utils module."""
from sklearn.utils.validation import check_is_fitted

from deepchecks.base import ModelWrapper, Dataset

def predict_dataset(dataset: Dataset, model):
    if isinstance(model, ModelWrapper):
        return model.predict_dataset(dataset)
    return model.predict(dataset.features_columns)

def predict_proba_dataset(dataset: Dataset, model):
    if isinstance(model, ModelWrapper):
        return model.predict_proba_dataset(dataset)
    return model.predict_proba(dataset.features_columns)

def check_is_model_fitted(model):
    if isinstance(model, ModelWrapper):
        return check_is_fitted(model._original_model)
    return check_is_fitted(model)
