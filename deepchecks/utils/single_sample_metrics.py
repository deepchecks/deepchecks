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
"""Common metrics to calculate performance on single samples."""
from typing import List, Optional

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksValueError


def calculate_neg_mse_per_sample(labels, predictions, index=None) -> pd.Series:
    """Calculate negative mean squared error per sample."""
    if index is None and isinstance(labels, pd.Series):
        index = labels.index
    return pd.Series([-(y - y_pred) ** 2 for y, y_pred in zip(labels, predictions)], index=index)


def calculate_neg_cross_entropy_per_sample(labels, probas: np.ndarray,
                                           model_classes: Optional[List] = None,
                                           index=None, is_multilabel: bool = False, eps=1e-15) -> pd.Series:
    """Calculate negative cross entropy per sample."""
    if not is_multilabel:
        if index is None and isinstance(labels, pd.Series):
            index = labels.index

        # transform categorical labels into integers
        if model_classes is not None:
            if any(x not in model_classes for x in labels):
                raise DeepchecksValueError(
                    f'Label observed values {sorted(np.unique(labels))} contain values '
                    f'that are not found in the model classes: {model_classes}.')
            if probas.shape[1] != len(model_classes):
                raise DeepchecksValueError(
                    f'Predicted probabilities shape {probas.shape} does not match the number of classes found in'
                    f' the labels: {model_classes}.')
            labels = pd.Series(labels).apply(list(model_classes).index)

        num_samples, num_classes = probas.shape
        one_hot_labels = np.zeros((num_samples, num_classes))
        one_hot_labels[list(np.arange(num_samples)), list(labels)] = 1
    else:
        one_hot_labels = labels

    return pd.Series(np.sum(one_hot_labels * np.log(probas + eps), axis=1), index=index)
