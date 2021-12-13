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
from deepchecks.base import ModelWrapper, Dataset

def predict_dataset(dataset: Dataset, model):
    """runs the predict function of the model on a given dataset features

    Args:
        dataset (Dataset): dataset to predict
        model: model that runs the prediction
    
    Returns:
        The prediction result
    """
    if isinstance(model, ModelWrapper):
        return model.predict_dataset(dataset)
    return model.predict(dataset.features_columns)

def predict_proba_dataset(dataset: Dataset, model):
    """runs the predict proba function of the model on a given dataset features

    Args:
        dataset (Dataset): dataset to predict
        model: model that runs the prediction
    
    Returns:
        The prediction result
    """
    if isinstance(model, ModelWrapper):
        return model.predict_proba_dataset(dataset)
    return model.predict_proba(dataset.features_columns)
