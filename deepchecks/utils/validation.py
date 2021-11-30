"""objects validation utilities."""
import typing as t
import sklearn
from deepchecks import errors


__all__ = ['model_type_validation']


def model_type_validation(model: t.Any):
    """Receive any object and check if it's an instance of a model we support.

    Raises:
        DeepchecksValueError: If the object is not of a supported type
    """
    supported_by_class_name = ('CatBoostClassifier', 'CatBoostRegressor')
    supported_by_class_instance = (sklearn.base.BaseEstimator,)
    if (
        not isinstance(model, supported_by_class_instance)
        and model.__class__.__name__ not in supported_by_class_name
    ):
        raise errors.DeepchecksValueError(
            'Model must inherit from one of supported '
            'models: sklearn.base.BaseEstimator or CatBoost'
        )
