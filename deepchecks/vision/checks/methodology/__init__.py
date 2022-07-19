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
"""Module containing the methodology checks in the vision package.

.. deprecated:: 0.7.0
        `deepchecks.vision.checks.distribution is deprecated and will be removed in deepchecks 0.8 version.
        Use `deepchecks.vision.checks.train_test_validation` instead.
"""
import warnings

from ..train_test_validation import PropertyLabelCorrelationChange, SimilarImageLeakage

__all__ = [
    'PropertyLabelCorrelationChange',
    'SimilarImageLeakage'
]

warnings.warn(
    'deepchecks.vision.checks.methodology is deprecated. Use deepchecks.vision.checks.train_test_validation instead.',
    DeprecationWarning
)
