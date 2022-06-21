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
import textwrap
import typing as t
from functools import wraps

from deepchecks.utils.logger import get_logger

__all__ = ['Substitution', 'Appender', 'deprecate_kwarg']


F = t.TypeVar('F', bound=t.Callable[..., t.Any])


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html


class Substitution:
    """Substitution docstring placeholders.

    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise AssertionError('Only positional or keyword args are allowed')

        self.params = args or kwargs

    def __call__(self, func: F) -> F:
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs) -> None:
        """Update self.params with supplied args."""
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


class Appender:
    r"""Append addendum to the docstring.

    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: t.Optional[str]

    def __init__(self, addendum: t.Optional[str], join: str = '', indents: int = 0):
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: F) -> F:
        func.__doc__ = func.__doc__ if func.__doc__ else ''
        self.addendum = self.addendum if self.addendum else ''
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = textwrap.dedent(self.join.join(docitems))
        return func


def indent(text: t.Optional[str], indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ''
    jointext = ''.join(['\n'] + ['    '] * indents)
    return jointext.join(text.split('\n'))


def deprecate_kwarg(
    old_name: str,
    new_name: t.Optional[str] = None,
) -> t.Callable[[F], F]:
    """Decorate a function with deprecated kwargs.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : Optional[str], default None
        Name of preferred argument in function.
    """
    def _deprecate_kwarg(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> t.Callable[..., t.Any]:
            if old_name in kwargs and new_name in kwargs:
                raise TypeError(
                    f'Can only specify {repr(old_name)} '
                    f'or {repr(new_name)}, not both'
                )
            elif old_name in kwargs and new_name is None:
                get_logger().warning(
                    f'the {repr(old_name)} keyword is deprecated and '
                    'will be removed in a future version. Please take '
                    f'steps to stop the use of {repr(old_name)}'
                )
            elif old_name in kwargs and new_name is not None:
                get_logger().warning(
                    f'the {repr(old_name)} keyword is deprecated, '
                    f'use {repr(new_name)} instead'
                )
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)
        return t.cast(F, wrapper)
    return _deprecate_kwarg
