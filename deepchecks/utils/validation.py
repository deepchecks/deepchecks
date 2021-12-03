"""objects validation utilities."""
import typing as t
import sklearn
from deepchecks import base # pylint: disable=unused-import, is used in type annotations
from deepchecks import errors


__all__ = ['model_type_validation', 'validate_model']


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


def validate_model(dataset: 'base.Dataset', model: t.Any):
    """Check model is able to predict on the dataset.

    Raise:
        DeepchecksValueError: if dataset does not match model
    """
    model_type_validation(model)

    error_message = (
        'In order to evaluate model correctness we need not empty dataset '
        'with the same set of features that was used to fit the model! {0}'
    )

    features = dataset.features_columns()
    features_names = set(dataset.features())
    model_features = getattr(model, 'feature_names_in_', None)

    if features is None:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received dataset without feature columns!'
        ))

    if len(features) == 0:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received empty dataset!'
        ))

    try:
        model_features = set(model_features) # type: ignore
        if model_features != features_names:
            raise errors.DeepchecksValueError(error_message.format(
                'But function received dataset with a different set of features!'
            ))
    except (TypeError, ValueError):
        # in case if 'model.feature_names_in_' was None or not iterable
        pass

    try:
        model.predict(features.head(1))
    except Exception as exc:
        raise errors.DeepchecksValueError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )
