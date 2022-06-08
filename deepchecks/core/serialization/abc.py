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
# pylint: disable=unnecessary-ellipsis
"""Main serialization abstractions."""
import abc
import io
import typing as t

import pandas as pd
from ipywidgets.widgets import Widget
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from typing_extensions import Protocol, runtime_checkable

from deepchecks.core import check_result as check_types  # pylint: disable=unused-import
from deepchecks.core.serialization import common

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
    'ABCDisplayItemsHandler'
]


T = t.TypeVar('T')


@runtime_checkable
class Serializer(Protocol[T]):
    """Base protocol for all other serializers."""

    value: T

    def __init__(self, value: T, **kwargs):
        self.value = value


@runtime_checkable
class HtmlSerializer(Serializer[T], Protocol):
    """To html serializer protocol."""

    def serialize(self, **kwargs) -> str:
        """Serialize into html."""
        ...


@runtime_checkable
class JsonSerializer(Serializer[T], Protocol):
    """To json serializer protocol."""

    def serialize(self, **kwargs) -> t.Union[t.Dict[t.Any, t.Any], t.List[t.Any]]:
        """Serialize into json."""
        ...


@runtime_checkable
class WidgetSerializer(Serializer[T], Protocol):
    """To ipywidget serializer protocol."""

    def serialize(self, **kwargs) -> Widget:
        """Serialize into ipywidgets.Widget instance."""
        ...


@runtime_checkable
class WandbSerializer(Serializer[T], Protocol):
    """To wandb metadata serializer protocol."""

    def serialize(self, **kwargs) -> t.Dict[str, 'WBValue']:
        """Serialize into Wandb media format."""
        ...


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


@runtime_checkable
class IPythonSerializer(Serializer[T], Protocol):
    """To IPython formatters list serializer."""

    def serialize(self, **kwargs) -> t.List[IPythonFormatter]:
        """Serialize into a list of objects that are Ipython displayable."""
        ...


class ABCDisplayItemsHandler(Protocol):
    """Trait that describes 'CheckResult.dislay' processing logic."""

    SUPPORTED_ITEM_TYPES = frozenset([
        str, pd.DataFrame, Styler, BaseFigure, t.Callable
    ])

    @classmethod
    @abc.abstractmethod
    def handle_display(
        cls,
        display: t.List['check_types.TDisplayItem'],
        **kwargs
    ) -> t.List[t.Any]:
        """Serialize list of display items.

        Parameters
        ----------
        display : List[Union[str, DataFrame, Styler, BaseFigure, Callable]]
            list of display items

        Returns
        -------
        List[Any]
        """
        return [cls.handle_item(it, index, **kwargs) for index, it in enumerate(display)]

    @classmethod
    @abc.abstractmethod
    def handle_item(cls, item: 'check_types.TDisplayItem', index: int, **kwargs) -> t.Any:
        """Serialize display item."""
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

    @classmethod
    @abc.abstractmethod
    def handle_string(cls, item: str, index: int, **kwargs) -> t.Any:
        """Handle textual item."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def handle_dataframe(cls, item: t.Union[pd.DataFrame, Styler], index: int, **kwargs) -> t.Any:
        """Handle dataframe item."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def handle_callable(cls, item: t.Callable, index: int, **kwargs) -> t.List[io.BytesIO]:
        """Handle callable."""
        # TODO: callable is a special case, add comments
        with common.switch_matplot_backend('agg'):
            item()
            return common.read_matplot_figures()

    @classmethod
    @abc.abstractmethod
    def handle_figure(cls, item: BaseFigure, index: int, **kwargs) -> t.Any:
        """Handle plotly figure item."""
        raise NotImplementedError()
