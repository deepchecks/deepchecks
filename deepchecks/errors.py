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


__all__ = ['DeepchecksBaseError', 'DeepchecksValueError']


class DeepchecksBaseError(Exception):
    """Base exception class for all 'Deepchecks' error types."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DeepchecksValueError(DeepchecksBaseError):
    """Exception class that represent a fault parameter was passed to Deepchecks."""

    pass
