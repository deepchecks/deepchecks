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
"""Module containing common (shared) functionality among core modules."""
import importlib
import typing as t

from typing_extensions import Literal as L

from deepchecks import __version__
from deepchecks.utils.logger import get_logger

__all__ = ['importable_name', 'import_type', 'validate_config']


def importable_name(obj: t.Any) -> t.Tuple[str, str]:
    """Return the full import name of an object type."""
    kind = type(obj) if not isinstance(obj, type) else obj
    name = kind.__qualname__
    module = kind.__module__
    return module, name


def import_type(
    module_name: str,
    type_name: str,
    base: t.Optional[t.Type[t.Any]] = None
) -> t.Type[t.Any]:
    """Import and return type instance by name."""
    module = importlib.import_module(module_name)
    type_ = getattr(module, type_name, None)

    if not isinstance(type_, type):
        name = type(type_).__qualname__  # type: ignore
        raise TypeError(f'Expected to import type instance, instead get - {name}')

    if base is not None and not issubclass(type_, base):
        name = type(type_).__qualname__  # type: ignore
        bname = type(base).__qualname__
        raise TypeError(f'Expected to import a subtype of "{bname}", instead got - {name}')

    return type_


VersionUnmatchAction = t.Union[L['raise'], L['warn'], None]  # noqa


def validate_config(
    conf: t.Dict[str, t.Any],
    version_unmatch: VersionUnmatchAction = 'warn'
) -> t.Dict[str, t.Any]:
    """Validate check/suite configuration dictionary."""
    if 'module_name' not in conf or not conf['module_name']:
        raise ValueError('Configuration must contain not empty "module_name" key of type string')

    if 'class_name' not in conf or not conf['class_name']:
        raise ValueError('Configuration must contain not empty "class_name" key of type string')

    if 'version' not in conf or not conf['version']:
        if version_unmatch == 'raise':
            raise ValueError(
                'Configuration must contain not emtpy '
                '"version" key of type string'
            )
        elif version_unmatch == 'warn':
            get_logger().warning(
                'Configuration was expected to contain not emtpy '
                '"version" key of type string'
            )

    elif conf['version'] != __version__:
        if version_unmatch == 'raise':
            raise ValueError(
                'Configuration was formed by different version of deepchecks package.\n'
                f'Configuration version: {conf["version"]}\n'
                f'Deepchecks version: {__version__}\n'
            )
        elif version_unmatch == 'warn':
            get_logger().warning(
                'Configuration was formed by different version of deepchecks package.\n'
                'Therefore a behavior of the check might be different than expected.\n'
                'Configuration version: %s\n'
                'Deepchecks version: %s\n',
                conf['version'],
                __version__
            )

    return conf
