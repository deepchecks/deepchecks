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
from typing import Any, Callable, Dict

__all__ = ['run_available_kwargs', 'initvars']


def run_available_kwargs(func: Callable, **kwargs):
    """Run the passed object only with available kwargs."""
    avail_kwargs = list(signature(func).parameters.keys())
    pass_kwargs = {}
    for kwarg_name in avail_kwargs:
        if kwarg_name in kwargs:
            pass_kwargs[kwarg_name] = kwargs[kwarg_name]
    return func(**pass_kwargs)


def initvars(
    obj: object,
    show_defaults: bool = False
) -> Dict[Any, Any]:
    """Return object __dict__ variables that was passed throw constructor (__init__ method).

    Parameters
    ----------
    obj : object
    show_defaults : bool, default False
        wherether to include vars with default value or not

    Returns
    -------
    Dict[Any, Any] subset of the obj __dict__
    """
    assert hasattr(obj, '__init__')
    params = signature(obj.__init__).parameters

    if show_defaults is True:
        return {
            k: v
            for k, v in vars(obj).items()
            if k in params
        }
    return {
        k: v
        for k, v in vars(obj).items()
        if k in params and v != params[k].default
    }
