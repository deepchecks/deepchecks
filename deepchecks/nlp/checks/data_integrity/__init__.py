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
"""Module importing all nlp checks."""

from .conflicting_labels import ConflictingLabels
from .frequent_substrings import FrequentSubstrings
from .property_label_correlation import PropertyLabelCorrelation
from .special_characters import SpecialCharacters
from .text_duplicates import TextDuplicates
from .text_property_outliers import TextPropertyOutliers
from .under_annotated_segments import UnderAnnotatedMetaDataSegments, UnderAnnotatedPropertySegments
from .unknown_tokens import UnknownTokens

__all__ = [
    'PropertyLabelCorrelation',
    'TextPropertyOutliers',
    'TextDuplicates',
    'ConflictingLabels',
    'SpecialCharacters',
    'UnknownTokens',
    'UnderAnnotatedMetaDataSegments',
    'UnderAnnotatedPropertySegments',
    'FrequentSubstrings',
]
