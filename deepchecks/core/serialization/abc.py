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
import abc

import pandas as pd
from pandas.io.formats.style import Styler
from ipywidgets.widgets import Widget
from wandb.sdk.data_types import WBValue
from plotly.basedatatypes import BaseFigure

from deepchecks.core.check_result import TDisplayItem


__all__ = [
    'Serializer',
    'HtmlSerializer',
    'JsonSerializer',
    'WidgetSerializer',
    'WandbSerializer',
    'ABCDisplayItemsHandler'
]


T = t.TypeVar("T")

@t.runtime_checkable
class Serializer(t.Protocol[T]):
    """Base protocol for all other serializers."""

    value: T

    def __init__(self, value: T, **kwargs):
        self.value = value


@t.runtime_checkable
class HtmlSerializer(Serializer[T], t.Protocol):
    """To html serializer protocol."""

    def serialize(self, **kwargs) -> str:
        ...


@t.runtime_checkable
class JsonSerializer(Serializer[T], t.Protocol):
    """To json serializer protocol."""

    def serialize(self, **kwargs) -> t.Union[t.Dict[t.Any, t.Any], t.List[t.Any]]:
        ...


@t.runtime_checkable
class WidgetSerializer(Serializer[T], t.Protocol):
    """To ipywidget serializer protocol."""

    def serialize(self, **kwargs) -> Widget:
        ...


@t.runtime_checkable
class WandbSerializer(Serializer[T], t.Protocol):
    """To wandb metadata serializer protocol."""

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        ...


class ABCDisplayItemsHandler(t.Protocol):

    SUPPORTED_ITEM_TYPES = frozenset([
        str, pd.DataFrame, Styler, BaseFigure, t.Callable
    ])

    @classmethod
    def handle_display(
        cls,
        display: t.List[TDisplayItem],
        **kwargs
    ) -> t.List[t.Any]:
        return [cls.handle_item(it, index) for index, it in enumerate(display)]

    @classmethod
    def handle_item(cls, item: TDisplayItem, index: int, **kwargs) -> t.Any:
        if isinstance(item, str):
            return cls.handle_string(item, index, **kwargs)
        elif isinstance(item, (pd.DataFrame, Styler)):
            return cls.handle_dataframe(item, index, **kwargs)
        elif isinstance(item, BaseFigure):
            return cls.handle_figure(item, index, **kwargs)
        elif callable(item):
            return cls.handle_callable(item, index, **kwargs)
        else:
            raise TypeError(f'Unable to handle display item of type: {type(item)}')

    @abc.abstractclassmethod
    def handle_string(cls, item: str, index: int, **kwargs) -> t.Any:
        """Handle textual item."""
        raise NotImplementedError()

    @abc.abstractclassmethod
    def handle_dataframe(cls, item: t.Union[pd.DataFrame, Styler], index: int, **kwargs) -> t.Any:
        """Handle dataframe item."""
        raise NotImplementedError()

    @abc.abstractclassmethod
    def handle_callable(cls, item: t.Callable, index: int, **kwargs) -> t.Any:
        """Handle callable."""
        raise NotImplementedError()

    @abc.abstractclassmethod
    def handle_figure(cls, item: BaseFigure, index: int, **kwargs) -> t.Any:
        """Handle plotly figure item."""
        raise NotImplementedError()
