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

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
_stream_handler.setFormatter(_formatter)
_logger.addHandler(_stream_handler)  # for some reason kaggle needs it
_logger.setLevel(logging.INFO)


def get_logger() -> logging.Logger:
    """Retutn the deepchecks logger."""
    return _logger


def get_verbosity() -> int:
    """Return the deepchecks logger verbosity level.

    Same as doing logging.getLogger('deepchecks').getEffectiveLevel().
    """
    return _logger.getEffectiveLevel()


def set_verbosity(level: int):
    """Set the deepchecks logger verbosity level.

    Same as doing logging.getLogger('deepchecks').setLevel(level).
    Control the package wide log level and the progrees bars - progress bars are level INFO.

    Examples
    --------
    >>> import logging
    >>> import deepchecks

    >>> # will disable progress bars
    >>> deepchecks.set_verbosity(logging.WARNING)
    >>> # will disable also any warnings deepchecks print
    >>> deepchecks.set_verbosity(logging.ERROR)
    """
    _logger.setLevel(level)
