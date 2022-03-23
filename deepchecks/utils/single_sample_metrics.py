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
"""Common metrics to calculate performance on single samples."""
import numpy as np


def per_sample_binary_cross_entropy(y_true: np.array, y_pred: np.array):
    """Calculate binary cross entropy on a single sample."""
    y_true = np.array(y_true)
    return - (np.tile(y_true.reshape((-1, 1)), (1, y_pred.shape[1])) *
              np.log(y_pred + np.finfo(float).eps)).sum(axis=1)


def per_sample_mse(y_true, y_pred):
    """Calculate mean square error on a single value."""
    return (y_true - y_pred) ** 2
