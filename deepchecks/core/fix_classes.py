# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
# pylint: disable=import-outside-toplevel
"""Module containing the fix classes and methods."""
import abc
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from deepchecks.tabular.dataset import Dataset
from deepchecks.utils.typing import BasicModel

__all__ = [
    'FixResult',
    'FixMixin',
    'SingleDatasetCheckFixMixin',
    'TrainTestCheckFixMixin'
]


class FixResult:
    """Class to store the result of a fix."""

    def __init__(
            self,
            fixed_train: Optional[Dataset] = None,
            fixed_test: Optional[Dataset] = None,
            fixed_model: Optional[BasicModel] = None
    ):
        """Initialize the FixResult object."""
        self.fixed_train = fixed_train
        self.fixed_test = fixed_test
        self.fixed_model = fixed_model

    def __repr__(self):
        """Return the string representation of the object."""
        display_train = f'fixed_train: {self.fixed_train.__class__}' if self.fixed_train is not None else ''
        display_test = f'fixed_test: {self.fixed_test.__class__}' if self.fixed_test is not None else ''
        display_model = f'fixed_model: {self.fixed_model.__class__}' if self.fixed_model is not None else ''
        display = ', '.join([s for s in [display_train, display_test, display_model] if s])

        return f'FixResult({display})'

    def to_dataset(self):
        """Return the fixed dataset."""
        if self.fixed_train is not None and self.fixed_test is not None:
            raise ValueError('Cannot return a single dataset when both train and test are fixed.')

        if self.fixed_train is not None:
            return self.fixed_train
        if self.fixed_test is not None:
            return self.fixed_test
        return None

    def to_datasets(self):
        """Return the fixed train and test datasets."""
        if self.fixed_train is None or self.fixed_test is None:
            raise ValueError('Cannot return both train and test datasets when only one is fixed.')
        return self.fixed_train, self.fixed_test


class FixMixin(abc.ABC):
    """Mixin for fixing functions."""

    def fix(self, dataset) -> FixResult:
        """Fix the user inputs."""
        raise NotImplementedError()

    def _validate_check_result(self, check_result):
        """Validate the check result."""
        raise NotImplementedError()


class SingleDatasetCheckFixMixin(FixMixin):
    """Extend FixMixin to for SingleDataset checks."""

    def get_context(
            self,
            dataset: Union[Dataset, pd.DataFrame],
            model: Optional[BasicModel] = None,
            y_pred: Optional[np.ndarray] = None,
            y_proba: Optional[np.ndarray] = None,
            model_classes: Optional[List] = None,
            with_display: bool = True
    ):
        """Get Context for check."""
        assert self.context_type is not None

        context = self.context_type(  # pylint: disable=not-callable
            train=dataset,
            model=model,
            with_display=with_display,
            y_pred_train=y_pred,
            y_proba_train=y_proba,
            model_classes=model_classes
        )

        return context


class TrainTestCheckFixMixin(FixMixin):
    """Extend FixMixin to for TrainTest checks."""

    def get_context(
            self,
            train_dataset: Union[Dataset, pd.DataFrame],
            test_dataset: Union[Dataset, pd.DataFrame],
            model: Optional[BasicModel] = None,
            y_pred_train: Optional[np.ndarray] = None,
            y_pred_test: Optional[np.ndarray] = None,
            y_proba_train: Optional[np.ndarray] = None,
            y_proba_test: Optional[np.ndarray] = None,
            model_classes: Optional[List] = None,
            with_display: bool = True
    ):
        """Get Context for check."""
        assert self.context_type is not None

        context = self.context_type(  # pylint: disable=not-callable
            train=train_dataset,
            test=test_dataset,
            model=model,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_proba_train=y_proba_train,
            y_proba_test=y_proba_test,
            model_classes=model_classes,
            with_display=with_display
        )
        return context
