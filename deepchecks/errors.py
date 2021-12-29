# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module with all deepchecks error types."""


__all__ = ['DeepchecksValueError', 'DeepchecksNotSupportedError', 'NumberOfFeaturesLimitError']


class DeepchecksBaseError(Exception):
    """Base exception class for all 'Deepchecks' error types."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DeepchecksValueError(DeepchecksBaseError):
    """Exception class that represent a fault parameter was passed to Deepchecks."""

    pass


class DeepchecksNotSupportedError(DeepchecksBaseError):
    """Exception class that represent unsupported action in Deepchecks."""

    pass


class NumberOfFeaturesLimitError(DeepchecksBaseError):
    """Represents a situation when a dataset contains to much features to be used for calculation."""

    pass
