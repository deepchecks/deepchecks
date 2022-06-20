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
"""Module with usefull decorators."""
import typing as t
import warnings
from functools import wraps

__all__ = ['deprecate_kwarg']


F = t.TypeVar('F', bound=t.Callable[..., t.Any])


# deprecate_kwarg is from "pandas.io._decorators" module
# pandas license: https://github.com/pandas-dev/pandas/blob/main/LICENSE


def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: t.Optional[str],
    mapping: t.Union[t.Mapping[t.Any, t.Any], t.Callable[[t.Any], t.Any], None] = None,
    stacklevel: int = 2,
) -> t.Callable[[F], F]:
    """Decorate a function with deprecated kwargs.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.
    """
    if mapping is not None and not hasattr(mapping, 'get') and not callable(mapping):
        raise TypeError(
            'mapping from old to new argument values must be dict or callable!'
        )

    def _deprecate_kwarg(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> t.Callable[..., t.Any]:
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f'the {repr(old_arg_name)} keyword is deprecated and '
                        'will be removed in a future version. Please take '
                        f'steps to stop the use of {repr(old_arg_name)}'
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f'the {old_arg_name}={repr(old_arg_value)} keyword is '
                        'deprecated, use '
                        f'{new_arg_name}={repr(new_arg_value)} instead'
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f'the {repr(old_arg_name)} keyword is deprecated, '
                        f'use {repr(new_arg_name)} instead'
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f'Can only specify {repr(old_arg_name)} '
                        f'or {repr(new_arg_name)}, not both'
                    )
                    raise TypeError(msg)
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return t.cast(F, wrapper)

    return _deprecate_kwarg
