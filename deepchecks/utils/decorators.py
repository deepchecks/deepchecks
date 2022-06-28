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
INDENT = '    '


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html


class DocStr(str):
    """Subclass of string that adds several additional methods."""

    def dedent(self) -> 'DocStr':
        return DocStr(textwrap.dedent(self))

    def indent(self) -> 'DocStr':
        return DocStr(indent(self))

    def __format__(self, *args, **kwargs):
        if len(args) == 0:
            return super().__format__(*args, **kwargs)

        allowed_modifiers = {'dedent', 'indent'}
        identation_modifier = args[0]
        parts = identation_modifier.split('*')

        if len(parts) == 1 and parts[0] in allowed_modifiers:
            return getattr(self, parts[0])()
        elif len(parts) == 2 and parts[0].isnumeric() and parts[1] in allowed_modifiers:
            n_of_times = int(parts[0])
            modifier = parts[1]
            s = self
            for _ in range(n_of_times):
                s = getattr(s, modifier)()
            return s

        return super().__format__(*args, **kwargs)


class Substitution:
    """Substitution docstring placeholders.

    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a dictionary suitable
    for performing substitution; then decorate a suitable function with
    the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "{author} wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"
    """

    def __init__(self, **kwargs):
        self.params = {
            k: DocStr(v) if not isinstance(v, DocStr) else v
            for k, v in kwargs.items()
        }

    def __call__(self, func: F) -> F:
        """Decorate a function."""
        func.__doc__ = func.__doc__ and func.__doc__.format(**self.params)
        return func

    def update(self, **kwargs) -> None:
        """Update self.params with supplied args."""
        if isinstance(self.params, dict):
            self.params.update({
                k: DocStr(v) if not isinstance(v, DocStr) else v
                for k, v in kwargs.items()
            })


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
        """Decorate a function."""
        func.__doc__ = func.__doc__ if func.__doc__ else ''
        self.addendum = self.addendum if self.addendum else ''
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = textwrap.dedent(self.join.join(docitems))
        return func


def indent(
    text: t.Optional[str],
    indents: int = 1,
    prefix: bool = False
) -> str:
    if not text or not isinstance(text, str):
        return ''
    identation = ''.join((INDENT for _ in range(indents)))
    jointext = ''.join(('\n', identation))
    output = jointext.join(text.split('\n'))
    return output if prefix is False else f'{identation}{output}'


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
                    'the %s keyword is deprecated and '
                    'will be removed in a future version. Please take '
                    'steps to stop the use of %s',
                    repr(old_name),
                    repr(old_name)
                )
            elif old_name in kwargs and new_name is not None:
                get_logger().warning(
                    'the %s keyword is deprecated, '
                    'use %s instead',
                    repr(old_name),
                    repr(new_name)
                )
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)
        return t.cast(F, wrapper)
    return _deprecate_kwarg


def get_routine_name(it: t.Any) -> str:
    if hasattr(it, '__qualname__'):
        return it.__qualname__
    elif callable(it) or isinstance(it, type):
        return it.__name__
    else:
        return type(it).__name__
