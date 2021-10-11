from typing import Any

import sklearn
import catboost


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
