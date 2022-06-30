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
"""Module for confusion matrix counts metrics."""


def _calc_recall(tp: float, fp: float, fn: float) -> float:  # pylint: disable=unused-argument
    """Calculate recall for given matches and number of positives."""
    if tp + fn == 0:
        return -1
    rc = tp / (tp + fn)
    return rc


def _calc_precision(tp: float, fp: float, fn: float) -> float:
    """Calculate precision for given matches and number of positives."""
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    pr = tp / (tp + fp)
    return pr


def _calc_f1(tp: float, fp: float, fn: float) -> float:
    """Calculate F1 for given matches and number of positives."""
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    rc = tp / (tp + fn)
    pr = tp / (tp + fp)
    f1 = (2 * rc * pr) / (rc + pr)
    return f1


def _calc_fpr(tp: float, fp: float, fn: float) -> float:
    """Calculate FPR for given matches and number of positives."""
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    return fp / (tp + fn)


def _calc_fnr(tp: float, fp: float, fn: float) -> float:
    """Calculate FNR for given matches and number of positives."""
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 1
    return fn / (tp + fn)


AVAILABLE_EVALUTING_FUNCTIONS = {"recall": _calc_recall, "fpr": _calc_fpr,
                                 "fnr": _calc_fnr, "precision": _calc_precision, "f1": _calc_f1}
