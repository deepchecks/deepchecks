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
"""Module for calculating detection precision and recall."""
import typing as t

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class ClassificationScorer(Metric):
    """Abstract class to calculate average precision and recall for various vision tasks.

    Parameters
    ----------
    score_func: Callable, default: None
        Score function (or loss function) with signature `score_func(y_true, y_pred, **kwargs)`
    needs_proba: bool, default: False
        Whether score_func requires the probabilites or not.
    **kwargs
        Additional parameters to be passed to score_func.
    """

    def __init__(
        self,
        score_func: t.Callable,
        needs_proba: bool = False,
        **kwargs
    ):
        super(ClassificationScorer, self).__init__(device="cpu")

        self.score_func = score_func
        self.needs_proba = needs_proba
        self.kwargs = kwargs


    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        super().reset()
        self._y_pred = []
        self._y = []

    @reinit__is_reduced
    def update(self, output):
        """Update metric with batch of samples."""
        y_pred, y = output

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()
        else:
            y_pred = np.array(y_pred)
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        else:
            y = np.array(y)

        if not self.needs_proba:
            y_pred = np.argmax(y_pred, axis=-1)

        self._y_pred.append(y_pred)
        self._y.append(y)     

    @sync_all_reduce("_y_pred", "_y")
    def compute(self):
        """Compute metric value."""
        y_pred = np.concatenate(self._y_pred)
        y = np.concatenate(self._y)
        return self.score_func(y, y_pred, **self.kwargs)
