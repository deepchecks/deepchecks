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
"""Module for evalution functions."""
import numpy as np


def _calc_recall(matched: np.ndarray, n_positives: int) -> float:
    """Calculate recall for given matches and number of positives."""
    if n_positives == 0:
        return -1
    if len(matched):
        tp = np.sum(matched)
        rc = tp / n_positives
        return rc
    return 0


def _calc_precision(matched: np.ndarray, n_positives: int) -> float:
    """Calculate precision for given matches and number of positives."""
    if n_positives == 0:
        return -1
    if len(matched):
        tp = np.sum(matched)
        fp = np.sum(~matched)
        pr = tp / (tp + fp)
        return pr
    return 0


def _calc_f1(matched: np.ndarray, n_positives: int) -> float:
    """Calculate f1 for given matches and number of positives."""
    if n_positives == 0:
        return -1
    if len(matched):
        tp = np.sum(matched)
        fp = np.sum(~matched)
        rc = tp / n_positives
        pr = tp / (tp + fp)
        f1 = (2 * rc * pr) / (rc + pr)
        return f1
    return 0


def _calc_fpr(matched: np.ndarray, n_positives: int) -> float:
    """Calculate FPR for given matches and number of positives."""
    if n_positives == 0:
        return -1
    if len(matched):
        fp = np.sum(~matched)
        return fp / n_positives
    return 0


def _calc_fnr(matched: np.ndarray, n_positives: int) -> float:
    """Calculate FNR for given matches and number of positives."""
    if n_positives == 0:
        return -1
    if len(matched):
        tp = np.sum(matched)
        fn = n_positives - tp
        return fn / n_positives
    return 1


AVAILABLE_EVALUTING_FUNCTIONS = {"recall": _calc_recall, "fpr": _calc_fpr,
                                 "fnr": _calc_fnr, "precision": _calc_precision, "f1": _calc_f1}
