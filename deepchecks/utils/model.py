from deepchecks.base import ModelWrapper, Dataset

def predict_dataset(model, dataset: Dataset):
    if isinstance(model, ModelWrapper):
        return model.predict_dataset(dataset)
    return model.predict(dataset.features_columns)

def predict_proba_dataset(model, dataset: Dataset):
    if isinstance(model, ModelWrapper):
        return model.predict_proba_dataset(dataset)
    return model.predict_proba(dataset.features_columns)

def get_class_name(model):
    if isinstance(model, ModelWrapper):
        return model.model_class_name
    return model.__class__.__name__
