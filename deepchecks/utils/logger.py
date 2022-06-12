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
"""Contain functions for configuring the deepchecks logger."""
import logging

__all__ = ['get_logger', 'get_verbosity', 'set_verbosity']

_logger = logging.getLogger('deepchecks')
_logger.addHandler(logging.StreamHandler())  # for some reason kaggle needs it


def get_logger() -> logging.Logger:
    '''Retutn the deepchecks logger'''
    return _logger


def get_verbosity() -> int:
    '''Return the deepchecks logger verbosity level.'''
    return _logger.getEffectiveLevel()


def set_verbosity(level: int):
    '''Sets the deepchecks logger verbosity level.'''
    _logger.setLevel(level)
