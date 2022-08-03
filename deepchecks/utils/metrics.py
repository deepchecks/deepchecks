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
"""Utils module with methods for general metrics."""
from typing import Union

import numpy as np
from sklearn.metrics._scorer import _BaseScorer

__all__ = ['get_gain', 'get_scorer_name', 'averaging_mechanism']

from deepchecks.core.errors import DeepchecksValueError


def get_gain(base_score, score, perfect_score, max_gain):
    """Get gain between base score and score compared to the distance from the perfect score."""
    distance_from_perfect = perfect_score - base_score
    scores_diff = score - base_score
    if distance_from_perfect == 0:
        # If both base score and score are perfect, return 0 gain
        if scores_diff == 0:
            return 0
        # else base_score is better than score, return -max_gain
        return -max_gain
    ratio = scores_diff / distance_from_perfect
    if ratio < -max_gain:
        return -max_gain
    if ratio > max_gain:
        return max_gain
    return ratio


def get_scorer_name(scorer) -> str:
    """Get scorer name from a scorer."""
    if isinstance(scorer, str):
        return scorer[:scorer.index('_per_class')] if scorer.endswith('_per_class') else scorer
    if hasattr(scorer, '__name__'):
        return scorer.__name__
    if isinstance(scorer, _BaseScorer):
        return scorer._score_func.__name__  # pylint: disable=protected-access
    return type(scorer).__name__


def averaging_mechanism(averaging_method: str, scores_per_class, weights=None) -> Union[np.ndarray, float]:
    """Receive scores per class and averaging method and returns result based on averaging_method.

    Parameters
    ----------
    averaging_method : str, default: 'per_class'
        Determines which averaging method to apply, possible values are:
        'per_class': Return a np array with the scores for each class (sorted by class name).
        'binary': Returns the score for the positive class. Should be used only in binary classification cases.
        'macro': Returns the mean of scores per class.
        'weighted': Returns a weighted mean of scores based provided weights.
    scores_per_class : array-like of shape (n_samples, n_classes)
        The score of the metric per class when considering said class as the positive class and the remaining
        classes as the negative.
    weights : array-like of shape (n_samples,), default: None
        True labels. Only required for 'weighted' averaging method.

    Returns
    -------
    score : Union[np.ndarray, float]
        The score for the given metric.
    """
    if averaging_method == 'binary':
        if len(scores_per_class) != 2:
            raise DeepchecksValueError('Averaging method "binary" can only be used in binary classification.')
        return scores_per_class[1]
    elif averaging_method == 'per_class':
        return np.asarray(scores_per_class)
    elif averaging_method == 'macro':
        # Classes that did not appear in the data are not considered as part of macro averaging.
        return np.mean(scores_per_class) if weights is None else np.mean(scores_per_class[weights != 0])
    elif averaging_method == 'weighted':
        if weights is None:
            raise DeepchecksValueError('Weights are required in order to apply weighted averaging method.')
        return np.multiply(scores_per_class, weights).sum() / sum(weights)
    else:
        raise DeepchecksValueError(f'Unknown averaging {averaging_method}')
