from sklearn.base import BaseEstimator

from deepchecks import Dataset

class ModelWrapper:
    def __init__(self, model: BaseEstimator):
        self._model = model
        self._predicted_datasets = {}
        self._predicted_proba_datasets = {}
        self.feature_importance = None

    def predict(self, data):
        return self._model.predict(data)

    def predict_proba(self, data):
        return self._model.predict_proba(data)

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