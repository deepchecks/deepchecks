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
# pylint: disable=unused-argument
"""Main serialization abstractions."""
import abc
# import io
import typing as t

import pandas as pd
from ipywidgets.widgets import Widget
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from typing_extensions import Protocol, runtime_checkable

from deepchecks.core import check_result as check_types  # pylint: disable=unused-import

# from deepchecks.core.serialization import common

try:
    from wandb.sdk.data_types.base_types.wb_value import WBValue  # pylint: disable=unused-import
except ImportError:
    pass


__all__ = [
    'Serializer',
    'HtmlSerializer',
    'JsonSerializer',
    'WidgetSerializer',
    'WandbSerializer',
    'DisplayItemsSerializer'
]


T = t.TypeVar('T')
TInput = t.TypeVar('TInput')
TOutput = t.TypeVar('TOutput')


class Serializer(abc.ABC, t.Generic[TInput, TOutput]):
    """Base protocol for all other serializers."""

    value: TInput

    def __init__(self, value: TInput, **kwargs):
        self.value = value

    @abc.abstractmethod
    def serialize(self, **kwargs) -> TOutput:
        """Serialize into html."""
        raise NotImplementedError()


class HtmlSerializer(Serializer[TInput, str]):
    """To html serializer protocol."""


JsonSerializable = t.Union[t.Dict[str, t.Any], t.List[t.Any]]


class JsonSerializer(Serializer[TInput, JsonSerializable]):
    """To json serializer protocol."""


class WidgetSerializer(Serializer[TInput, Widget]):
    """To ipywidget serializer protocol."""


class WandbSerializer(Serializer[TInput, t.Dict[str, 'WBValue']]):
    """To wandb metadata serializer protocol."""


@runtime_checkable
class HTMLFormatter(Protocol):
    """An HTML formatter."""

    def _repr_html_(self) -> t.Any: ...


@runtime_checkable
class MarkdownFormatter(Protocol):
    """A Markdown formatter."""

    def _repr_markdown_(self) -> t.Any: ...


@runtime_checkable
class JSONFormatter(Protocol):
    """A JSON formatter."""

    def _repr_json_(self) -> t.Any: ...


@runtime_checkable
class JPEGFormatter(Protocol):
    """A JPEG formatter."""

    def _repr_jpeg_(self) -> t.Any: ...


@runtime_checkable
class PNGFormatter(Protocol):
    """A PNG formatter."""

    def _repr_png_(self) -> t.Any: ...


@runtime_checkable
class SVGFormatter(Protocol):
    """An SVG formatter."""

    def _repr_png_(self, **kwargs) -> t.Any: ...


@runtime_checkable
class IPythonDisplayFormatter(Protocol):
    """An Formatter for objects that know how to display themselves."""

    def _ipython_display_(self, **kwargs) -> t.Any: ...


@runtime_checkable
class MimeBundleFormatter(Protocol):
    """A Formatter for arbitrary mime-types."""

    def _repr_mimebundle_(self, **kwargs) -> t.Any: ...


# NOTE: For more info about IPython formatters API refer to the next documentation page:
# - https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.formatters.html


IPythonFormatter = t.Union[
    HTMLFormatter,
    MarkdownFormatter,
    JSONFormatter,
    JPEGFormatter,
    PNGFormatter,
    SVGFormatter,
    IPythonDisplayFormatter,
    MimeBundleFormatter
]


class IPythonSerializer(Serializer[TInput, t.List[IPythonFormatter]]):
    """To IPython formatters list serializer."""


DisplayItems = t.Sequence['check_types.TDisplayItem']
T = t.TypeVar('T')


class DisplayItemsSerializer(Serializer[DisplayItems, t.List[T]]):
    """Trait that describes 'CheckResult.dislay' processing logic."""

    @classmethod
    def supported_item_types(cls):
        """Return set of supported types of display items."""
        return frozenset([
            str, pd.DataFrame, Styler, BaseFigure, t.Callable, check_types.DisplayMap
        ])

    def serialize(self, **kwargs) -> t.List[T]:
        """Serialize display items."""
        return self.handle_display(self.value, **kwargs)

    def handle_display(self, display: DisplayItems, **kwargs) -> t.List[T]:
        """Serialize list of display items.

        Parameters
        ----------
        display : List[Union[str, DataFrame, Styler, BaseFigure, Callable]]
            list of display items

        Returns
        -------
        List[Any]
        """
        return [self.handle_item(it, index, **kwargs) for index, it in enumerate(display)]

    def handle_item(self, item: 'check_types.TDisplayItem', index: int, **kwargs) -> T:
        """Serialize display item."""
        if isinstance(item, str):
            return self.handle_string(item, index, **kwargs)
        elif isinstance(item, (pd.DataFrame, Styler)):
            return self.handle_dataframe(item, index, **kwargs)
        elif isinstance(item, BaseFigure):
            return self.handle_figure(item, index, **kwargs)
        elif isinstance(item, check_types.DisplayMap):
            return self.handle_display_map(item, index, **kwargs)
        elif callable(item):
            return self.handle_callable(item, index, **kwargs)
        else:
            raise TypeError(f'Unable to handle display item of type: {type(item)}')

    @abc.abstractmethod
    def handle_string(self, item: str, index: int, **kwargs) -> T:
        """Handle textual item."""
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_dataframe(self, item: t.Union[pd.DataFrame, Styler], index: int, **kwargs) -> T:
        """Handle dataframe item."""
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_callable(self, item: t.Callable[[], None], index: int, **kwargs) -> T:
        """Handle callable."""
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_figure(self, item: BaseFigure, index: int, **kwargs) -> T:
        """Handle plotly figure item."""
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_display_map(self, item: 'check_types.DisplayMap', index: int, **kwargs) -> T:
        """Handle display map instance item."""
        raise NotImplementedError()
