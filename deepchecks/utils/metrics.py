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
from sklearn.metrics._scorer import _BaseScorer

__all__ = ['get_gain', 'get_scorer_name']


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
