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
"""Module for classification datasets and models."""

from . import mnist_torch

__all__ = ['mnist_torch']

try:
    from . import mnist_tensorflow  # noqa: F401
except ImportError:
    pass
else:
    __all__.append('mnist_tensorflow')
