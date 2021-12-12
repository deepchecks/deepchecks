from sklearn.base import BaseEstimator

from deepchecks import Dataset
from deepchecks.utils.features import calculate_feature_importance

__all__ = ['ModelWrapper']


class ModelWrapper:
    def __init__(self, model: BaseEstimator, dataset = None):
        self._model = model
        self._predicted_datasets = {}
        self._predicted_proba_datasets = {}
        self.model_class_name = model.__class__.__name__

        if dataset:
            self.feature_importance = \
                calculate_feature_importance(model=model, dataset=dataset)
        else:
            self.feature_importance = None

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data, *args, **kwargs)

    def predict_proba(self, data, *args, **kwargs):
        return self._model.predict_proba(data, *args, **kwargs)

    def predict_dataset(self, dataset: Dataset):
        prediction = self._predicted_datasets.get(dataset)
        if prediction:
            return prediction
        prediction = self._model.predict(dataset.features_columns)
        self._predicted_datasets[dataset] = prediction
        return prediction

    def predict_proba_dataset(self, dataset: Dataset):
        prediction = self._predicted_proba_datasets.get(dataset)
        if prediction:
            return prediction
        prediction = self._model.predict_proba(dataset.features_columns)
        self._predicted_proba_datasets[dataset] = prediction
        return prediction
