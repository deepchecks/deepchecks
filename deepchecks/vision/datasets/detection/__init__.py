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
"""Module for detection datasets and models."""

from . import coco_torch, mask

__all__ = ['coco_torch', 'mask']

try:
    from . import coco_tensorflow  # noqa: F401
except ImportError:
    pass
else:
    __all__.append('coco_tensorflow')
