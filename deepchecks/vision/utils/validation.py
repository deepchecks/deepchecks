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
"""Module for validation of the vision module."""
import typing as t

from deepchecks.core import errors
from deepchecks import vision  # pylint: disable=unused-import, is used in type annotations


__all__ = ['validate_model']


def validate_model(dataset: 'vision.VisionDataset', model: t.Any):
    """Receive a dataset and a model and check if they are compatible.

    Parameters
    ----------
    dataset : VisionDataset
        Built on a dataloader on which the model can infer.
    model : Any
        Model to be validated

    Raises
    ------
    DeepchecksValueError
        If the dataset and the model are not compatible
    """
    try:
        model(next(iter(dataset.get_data_loader()))[0])
    except Exception as exc:
        raise errors.ModelValidationError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )
