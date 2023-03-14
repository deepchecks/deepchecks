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

from deepchecks.core.checks import DatasetKind
from deepchecks.tabular import deprecation_warnings  # pylint: disable=unused-import # noqa: F401
from deepchecks.tabular._shared_docs import docstrings
from deepchecks.tabular.context import Context
from deepchecks.tabular.dataset import Dataset
from deepchecks.utils.typing import BasicModel

__all__ = [
    'FixMixin',
    'SingleDatasetCheckFixMixin'
]


class FixMixin(abc.ABC):
    """Mixin for fixing functions."""

    def fix(self, *args, **kwargs):
        """Fix the user inputs."""
        raise NotImplementedError()

    @property
    def fix_params(self):
        """Return the fix params."""
        raise NotImplementedError()

    @property
    def problem_description(self):
        """Return a problem description."""
        raise NotImplementedError()

    @property
    def manual_solution_description(self):
        """Return a manual solution description."""
        raise NotImplementedError()

    @property
    def automatic_solution_description(self):
        """Return an automatic solution description."""
        raise NotImplementedError()


class SingleDatasetCheckFixMixin(FixMixin):
    """Extend FixMixin to for performance checks."""

    @docstrings
    def fix(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        model: Optional[BasicModel] = None,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        model_classes: Optional[List] = None,
        with_display: bool = True,
        **kwargs
    ) -> Dataset:
        """Run check.

        Parameters
        ----------
        dataset: Union[Dataset, pd.DataFrame]
            Dataset or DataFrame object, representing data an estimator was fitted on
        model: Optional[BasicModel], default: None
            A scikit-learn-compatible fitted estimator instance
        {additional_context_params:2*indent}
        """
        assert self.context_type is not None

        context = self.context_type(  # pylint: disable=not-callable
            train=dataset,
            model=model,
            with_display=with_display,
            y_pred_train=y_pred,
            y_proba_train=y_proba,
            model_classes=model_classes
        )
        updated_context = self.fix_logic(context, dataset_kind=DatasetKind.TRAIN, **kwargs)
        return updated_context.train

    @abc.abstractmethod
    def fix_logic(self, context, dataset_kind, **kwargs) -> Context:
        """Run check."""
        raise NotImplementedError()
