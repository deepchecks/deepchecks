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
from functools import lru_cache
from inspect import Signature, signature
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
    include_defaults: bool = False,
    include_kwargs: bool = False,
    include_properties: bool = False,
) -> Dict[Any, Any]:
    """Return object __dict__ variables that was passed throw constructor (__init__ method).

    Parameters
    ----------
    obj : object
    include_defaults : bool, default False
        wherether to include vars with default value or not

    Returns
    -------
    Dict[Any, Any] subset of the obj __dict__
    """
    assert hasattr(obj, '__init__')
    state = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    signature = extract_signature(obj.__init__)  # pylint: disable=redefined-outer-name
    bind = signature.bind(**state)

    if include_defaults is True:
        bind.apply_defaults()
        arguments = bind.arguments
    else:
        arguments = {
            k: v
            for k, v in bind.arguments.items()
            if signature.parameters[k].default != v
        }

    if not include_kwargs:
        arguments.pop('kwargs', None)

    if not include_properties:
        arguments.pop('properties_list', None)
        arguments.pop('image_properties', None)
        arguments.pop('label_properties', None)
        arguments.pop('prediction_properties', None)

    return arguments


@lru_cache(maxsize=None)
def extract_signature(obj: Callable[..., Any]) -> Signature:
    """Extract signature object from a callable instance.

    Getting a callable signature is a heavy and not cheap op
    therefore we are caching it.
    """
    return signature(obj)
