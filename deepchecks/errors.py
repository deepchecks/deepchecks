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


__all__ = [
    'DeepchecksValueError',
    'DeepchecksNotSupportedError',
    'DeepchecksProcessError',
    'NumberOfFeaturesLimitError',
    'DatasetValidationError',
    'ModelValidationError'
]


class DeepchecksBaseError(Exception):
    """Base exception class for all 'Deepchecks' error types."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DeepchecksValueError(DeepchecksBaseError):
    """Exception class that represent a fault parameter was passed to Deepchecks."""

    pass


class DeepchecksNotSupportedError(DeepchecksBaseError):
    """Exception class that represents an unsupported action in Deepchecks."""

    pass


class DeepchecksProcessError(DeepchecksBaseError):
    """Exception class that represents an issue with a process."""

    pass


class NumberOfFeaturesLimitError(DeepchecksBaseError):
    """Represents a situation when a dataset contains to much features to be used for calculation."""

    pass


class DeepchecksTimeoutError(DeepchecksBaseError):
    """Represents a situation when a computation takes too long and is interrupted."""

    pass


class ValidationError(DeepchecksBaseError):
    """Represents more specific case of the ValueError (DeepchecksValueError)."""

    pass


class DatasetValidationError(DeepchecksBaseError):
    """Represents unappropriate Dataset instance.

    Should be used in a situation when a routine (like check instance, utility function, etc)
    expected and received a dataset instance that did not meet routine requirements.
    """

    pass


class ModelValidationError(DeepchecksBaseError):
    """Represents unappropriate model instance.

    Should be used in a situation when a routine (like check instance, utility function, etc)
    expected and received a dataset instance that did not meet routine requirements.
    """

    pass
