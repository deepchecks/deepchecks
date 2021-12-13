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
"""ModelWrapper class module."""
import inspect
from sklearn.base import BaseEstimator

__all__ = ['ModelWrapper']


class ModelWrapper:
    """wrapper for a model that a suite receives."""

    def __init__(self, model: BaseEstimator):
        """Construct the ModelWrapper class.

        Args:
            model (BaseEstimator): sklearn compatible model
        """
        self._original_model = model
        self._predicted_datasets = {}
        self._predicted_proba_datasets = {}
        self.model_class_name = model.__class__.__name__
        self.feature_importance = None
        self.__class__.__name__ = model.__class__.__name__
        for n, m in inspect.getmembers(model):
            if not n.startswith('_'):
                setattr(self, n, m)

    @property
    def original_model(self):
        return self._original_model

    def predict_dataset(self, dataset: 'Dataset'):
        prediction = self._predicted_datasets.get(dataset)
        if prediction is not None:
            return prediction
        prediction = self._original_model.predict(dataset.features_columns)
        self._predicted_datasets[dataset] = prediction
        return prediction

    def predict_proba_dataset(self, dataset: 'Dataset'):
        prediction = self._predicted_proba_datasets.get(dataset)
        if prediction is not None:
            return prediction
        prediction = self._original_model.predict_proba(dataset.features_columns)
        self._predicted_proba_datasets[dataset] = prediction
        return prediction
