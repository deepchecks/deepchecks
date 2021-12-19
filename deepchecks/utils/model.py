# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""module for model functions utils."""
from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from deepchecks.errors import DeepchecksValueError


__all__ = ['get_model_of_pipeline']


def get_model_of_pipeline(model: Union[Pipeline, BaseEstimator]):
    """Return the model of a given Pipeline or itself if a BaseEstimator is given.

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
