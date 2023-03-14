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
# pylint: disable=import-outside-toplevel
"""Module containing the fix classes and methods."""
import abc
from typing import Dict, Optional

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.logger import get_logger

__all__ = [
    'FixMixin',
    'SingleDatasetCheckFixMixin'
]


class FixMixin(abc.ABC):
    """Mixin for fixing functions."""

    def fix(self, *args, **kwargs):
        """Fix the user inputs."""
        raise NotImplementedError()

    @property
    def problem_description(self):
        """Return a problem description."""
        raise NotImplementedError()

    @property
    def manual_solution_description(self):
        """Return a manual solution description."""
        raise NotImplementedError()

    @property
    def automatic_solution_description(self):
        """Return an automatic solution description."""
        raise NotImplementedError()


class SingleDatasetCheckFixMixin(FixMixin):
    """Extend FixMixin to for performance checks."""

    def fix(self, dataset, *args, **kwargs):
        """Fix the user inputs."""
        raise NotImplementedError()