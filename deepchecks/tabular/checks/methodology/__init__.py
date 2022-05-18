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
"""
Module contains checks for methodological flaws in the model building process.

.. deprecated:: 0.7.0
        :mod:`deepchecks.tabular.checks.methodology is deprecated and will be removed in deepchecks 0.8 version.
        Use :mod:`deepchecks.tabular.checks.model_evaluation`, :mod:`deepchecks.tabular.checks.train_test_validation`,
        :mod:`deepchecks.tabular.checks.integrity` instead.

"""
import warnings

from ..model_evaluation import BoostingOverfit, ModelInferenceTime, UnusedFeatures
from ..train_test_validation import (DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                     DateTrainTestLeakageOverlap, IdentifierLeakage, IndexTrainTestLeakage,
                                     SingleFeatureContributionTrainTest, TrainTestSamplesMix)
from ..integrity import SingleFeatureContribution


__all__ = [
    'BoostingOverfit',
    'UnusedFeatures',
    'SingleFeatureContribution',
    'SingleFeatureContributionTrainTest',
    'IndexTrainTestLeakage',
    'TrainTestSamplesMix',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLeakage',
    'ModelInferenceTime',
    'DatasetsSizeComparison',
]

warnings.warn(
                'deepchecks.tabular.checks.methodology is deprecated and will be removed in deepchecks 0.8 version. '
                'Use deepchecks.tabular.checks.model_evaluation, deepchecks.tabular.checks.train_test_validation,'
                'deepchecks.tabular.checks.integrity` instead.',
                DeprecationWarning
            )
