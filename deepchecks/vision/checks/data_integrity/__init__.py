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
"""Module containing the data integrity checks in the vision package."""
from .image_property_outliers import ImagePropertyOutliers
from .label_property_outliers import LabelPropertyOutliers

__all__ = [
    'ImagePropertyOutliers',
    'LabelPropertyOutliers'
]
