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

def get_class_name(model):
    if isinstance(model, ModelWrapper):
        return model.model_class_name
    return model.__class__.__name__

def check_is_model_fitted(model):
    if isinstance(model, ModelWrapper):
        return check_is_fitted(model.original_model)
    return check_is_fitted(model)
