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
from collections import defaultdict
from functools import wraps

from deepchecks.utils.logger import get_logger

__all__ = ['Substitution', 'Appender', 'deprecate_kwarg', 'ParametersCombiner']


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


def scrap_parameters_lines(docstring: str) -> t.List[str]:
    """Return docstring parameters section lines."""
    lines = docstring.split('\n')
    n_of_lines = len(lines)
    output = []
    if n_of_lines > 2:
        parameters_section = 'Parameters'
        section_underline = {'-'}
        end_of_parameters = {'', 'Returns', 'Examples', 'See also', 'Notes', 'Raises'}
        l1, l2 = lines[0].strip(), lines[1].strip()
        current_line = 0
        while current_line < n_of_lines:
            if l1 == parameters_section and set(l2) == section_underline:
                while current_line < n_of_lines:
                    l = lines[current_line]
                    ls = l.strip()
                    if ls in end_of_parameters or set(ls) == section_underline:
                        return output
                    output.append(l)
                    current_line += 1
            else:
                l1 = l2
                l2 = lines[current_line].strip()
                current_line += 1
    return output


Parameter = t.Tuple[str, str, str]  # name, type-desc, desc


def parse_parameters_section(docstring: str) -> t.Tuple[Parameter, ...]:
    """Parse docstring parameters section."""
    parameters_section = scrap_parameters_lines(docstring)

    if not parameters_section:
        return tuple()

    n_of_lines = len(parameters_section)
    current_line = 0
    parameter_line = parameters_section[0]
    level = len(parameter_line) - len(parameter_line.lstrip(' \t'))
    type_and_name_devider = ':'
    output = []

    while current_line < n_of_lines:
        parameter_line = parameters_section[current_line]

        if type_and_name_devider not in parameter_line:
            # Unknown structure, exit
            return tuple(output)

        name, type_desc = parameter_line.split(type_and_name_devider, maxsplit=1)
        name, type_desc = name.strip(), type_desc.strip()
        description = []
        current_line += 1

        while current_line < n_of_lines:
            desc_line = parameters_section[current_line]
            desc_line_level = len(desc_line) - len(desc_line.lstrip(' \t'))
            if desc_line_level == level:
                break
            elif desc_line_level > level:
                description.append(desc_line.strip())
                current_line += 1
            else:
                # Unknown structure, exit
                return tuple(output)

        output.append((name, type_desc, '\n'.join(description)))

    return tuple(output)


class ParametersCombiner:
    """Combine docstring parameters from two or more routines."""

    __slots__ = ('routines', 'template_arg_name', '_parameters', '_combined_parameters')

    def __init__(
        self,
        *routines: t.Any,
        template_arg_name: str = 'combined_parameters'
    ):
        self.routines = routines
        self.template_arg_name = template_arg_name
        self._parameters = None
        self._combined_parameters = None

    @property
    def parameters(self) -> t.Tuple[
        t.Tuple[object, t.Tuple[Parameter, ...]],
        ...
    ]:
        """Return collected routines parameters."""
        if self._parameters is not None:
            return self._parameters
        else:
            parameters = []
            docstring_parameters_attr = '__docstring_parameters__'
            for it in self.routines:
                if not hasattr(it, '__doc__'):
                    raise AttributeError(
                        f'element {get_routine_name(it)} does not have '
                        'documentation string'
                    )
                if hasattr(it, docstring_parameters_attr):
                    parameters.append((it, getattr(it, docstring_parameters_attr)))
                else:
                    docstring_parameters = parse_parameters_section(it.__doc__ or '')
                    setattr(it, docstring_parameters_attr, docstring_parameters)
                    parameters.append((it, docstring_parameters))
            self._parameters = tuple(parameters)
            return self._parameters

    @property
    def combined_parameters(self) -> str:
        """Return combined parameters docstring."""
        if self._combined_parameters is not None:
            return self._combined_parameters
        else:
            parameters_consumers = defaultdict(set)
            parameters = {}
            ignore = {'**kwargs', '*args'}

            for routine, params in self.parameters:
                for name, type_desc, desc in params:
                    if name in ignore:
                        continue
                    parameters_consumers[name].add(get_routine_name(routine))
                    parameters[name] = (type_desc, desc)

            parameter_template = '{} : {}\n{}'
            description_template = '{}\n(Is used by: {})'
            lines = []

            for name, (type_desc, desc) in parameters.items():
                used_by = ', '.join(parameters_consumers[name])
                desc = description_template.format(desc, used_by)
                desc = indent(desc, indents=1, prefix=True)
                lines.append(parameter_template.format(name, type_desc, desc))

            self._combined_parameters = '\n'.join(lines)
            return self._combined_parameters

    def __call__(self, routine: F) -> F:
        """Decorate a routine."""
        if not hasattr(routine, '__doc__'):
            raise AttributeError(
                f'routine {get_routine_name(routine)} does not have '
                'documentation string'
            )
        template_args = {self.template_arg_name: DocStr(self.combined_parameters)}
        routine.__doc__ = routine.__doc__ and routine.__doc__.format(**template_args)
        return routine


def get_routine_name(it: t.Any) -> str:
    if hasattr(it, '__qualname__'):
        return it.__qualname__
    elif callable(it) or isinstance(it, type):
        return it.__name__
    else:
        return type(it).__name__
