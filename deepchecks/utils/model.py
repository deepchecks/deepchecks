# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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

from deepchecks.utils.typing import BasicModel

__all__ = ['get_model_of_pipeline']


def get_model_of_pipeline(model: Union[Pipeline, BasicModel]):
    """Return the model of a given Pipeline or itself if a BaseEstimator is given.

    Parameters
    ----------
    model : Union[Pipeline, BasicModel]
        a Pipeline or a BasicModel

    Returns
    -------
    Union[Pipeline, BasicModel]
        the inner BaseEstimator of the Pipeline or itself
    """
    if isinstance(model, Pipeline):
        # get model type from last step in pipeline
        return model.steps[-1][1]
    return model
