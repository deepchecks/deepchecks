from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from deepchecks.errors import DeepchecksValueError


__all__ = ['get_model_of_pipeline']


def get_model_of_pipeline(model: Union[Pipeline, BaseEstimator]):
    """return the model of a given Pipeline or itself if a BaseEstimator is given.

    Args:
        model (Union[Pipeline, BaseEstimator]): a Pipeline or a BaseEstimator model
    Returns:
        the inner BaseEstimator of the Pipeline or itself
    """
    if isinstance(model, Pipeline):
        # get feature importance from last model in pipeline
        internal_estimator_list = [x[1] for x in model.steps if isinstance(x[1], BaseEstimator)]
        if internal_estimator_list:
            return internal_estimator_list[-1]
        raise DeepchecksValueError('Recived a pipeline without an sklearn compatible model')
    return model
