from typing import Any
import sklearn
import catboost

__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation']

from pandas import DataFrame

from mlchecks import Dataset

SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]


class MLChecksValueError(ValueError):
    pass


def model_type_validation(model: Any):
    """Receive any object and check if it's an instance of a model we support

    Raises
        MLChecksException: If the object is not of a supported type
    """
    if not any([isinstance(model, base) for base in SUPPORTED_BASE_MODELS]):
        raise MLChecksValueError(f'Model must inherit from one of supported models: {SUPPORTED_BASE_MODELS}')


def validate_dataset(ds) -> Dataset:
    """
    Receive an object and throws error if it's not pandas DataFrame or MLChecks Dataset.
    Also returns the object as MLChecks Dataset

    Returns
        (Dataset): object converted to MLChecks dataset
    """
    if isinstance(ds, Dataset):
        return ds
    elif isinstance(ds, DataFrame):
        return Dataset(ds)
    else:
        raise MLChecksValueError(f"dataset must be of type DataFrame or Dataset instead got: "
                                 f"{type(ds).__name__}")
