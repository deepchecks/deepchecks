from sklearn.base import BaseEstimator

__all__ = ['ModelWrapper']


class ModelWrapper:
    def __init__(self, model: BaseEstimator):
        self.original_model = model
        self._predicted_datasets = {}
        self._predicted_proba_datasets = {}
        self.model_class_name = model.__class__.__name__
        self.model_features = getattr(model, 'feature_names_in_', None)
        self.feature_importance = None

    def predict(self, data, *args, **kwargs):
        return self.original_model.predict(data, *args, **kwargs)

    def predict_proba(self, data, *args, **kwargs):
        return self.original_model.predict_proba(data, *args, **kwargs)

    def predict_dataset(self, dataset: 'Dataset'):
        prediction = self._predicted_datasets.get(dataset)
        if prediction is not None:
            return prediction
        prediction = self.original_model.predict(dataset.features_columns)
        self._predicted_datasets[dataset] = prediction
        return prediction

    def predict_proba_dataset(self, dataset: 'Dataset'):
        prediction = self._predicted_proba_datasets.get(dataset)
        if prediction is not None:
            return prediction
        prediction = self.original_model.predict_proba(dataset.features_columns)
        self._predicted_proba_datasets[dataset] = prediction
        return prediction
