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
"""Module containing the train test validation check in the nlp package."""

from .keyword_frequency_drift import KeywordFrequencyDrift
from .train_test_label_drift import TrainTestLabelDrift
from .text_embeddings_drift import TextEmbeddingsDrift

__all__ = ['KeywordFrequencyDrift',
           'TrainTestLabelDrift',
           'TextEmbeddingsDrift']
