# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
