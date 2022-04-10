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
"""Contain functions for handling function in checks."""
from inspect import signature
from typing import Callable


__all__ = ['run_available_kwargs']


def run_available_kwargs(func: Callable, **kwargs):
    """Run the passed object only with available kwargs."""
    avail_kwargs = list(signature(func).parameters.keys())
    pass_kwargs = {}
    for kwarg_name in avail_kwargs:
        if kwarg_name in kwargs:
            pass_kwargs[kwarg_name] = kwargs[kwarg_name]
    return func(**pass_kwargs)
