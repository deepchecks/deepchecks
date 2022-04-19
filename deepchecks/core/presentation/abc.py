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
"""Main serialization abstractions."""
import typing as t
from ipywidgets.widgets import Widget
from wandb.sdk.data_types import WBValue


__all__ = [
    'Serializer',
    'HtmlSerializer',
    'JsonSerializer',
    'WidgetSerializer',
    'WandbSerializer'
]


T = t.TypeVar("T")


class Serializer(t.Protocol[T]):
    """Base protocol for all other serializers."""

    value: T

    def __init__(self, value: T, **kwargs):
        self.value = value


class HtmlSerializer(Serializer[T]):
    """To html serializer protocol."""

    def serialize(self, **kwargs) -> str:
        ...


class JsonSerializer(Serializer[T]):
    """To json serializer protocol."""

    def serialize(self, **kwargs) -> t.Union[t.Dict[t.Any, t.Any], t.List[t.Any]]:
        ...


class WidgetSerializer(Serializer[T]):
    """To ipywidget serializer protocol."""

    def serialize(self, **kwargs) -> Widget:
        ...


class WandbSerializer(Serializer[T]):
    """To wandb metadata serializer protocol."""

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        ...
