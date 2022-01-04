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


__all__ = ['get_model_of_pipeline']


def get_model_of_pipeline(model: Union[Pipeline, BaseEstimator]):
    """Return the model of a given Pipeline or itself if a BaseEstimator is given.

    Args:
        model (Union[Pipeline, BaseEstimator]): a Pipeline or a BaseEstimator model
    Returns:
        the inner BaseEstimator of the Pipeline or itself
    """
    if isinstance(model, Pipeline):
        # get model type from last step in pipeline
        return model.steps[-1][1]
    return model
