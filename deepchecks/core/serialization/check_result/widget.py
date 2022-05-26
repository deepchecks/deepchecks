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
"""Module containing ipywidget serializer for the CheckResult type."""
import typing as t

import pandas as pd
import plotly.graph_objects as go
from ipywidgets import HTML, VBox, Widget
from plotly.basedatatypes import BaseFigure

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.common import normalize_widget_style

from . import html

__all__ = ['CheckResultSerializer']


class CheckResultSerializer(WidgetSerializer['check_types.CheckResult']):
    """Serializes any CheckResult instance into ipywidgets.Widget instance.

    Parameters
    ----------
    value : CheckResult
        CheckResult instance that needed to be serialized.
    """

    def __init__(self, value: 'check_types.CheckResult', **kwargs):
        if not isinstance(value, check_types.CheckResult):
            raise TypeError(
                f'Expected "CheckResult" but got "{type(value).__name__}"'
            )
        self.value = value
        self._html_serializer = html.CheckResultSerializer(self.value)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[html.CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        """Serialize a CheckResult instance into ipywidgets.Widget instance.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into theoutput,
            in case of 'None' all sections will be included

        Returns
        -------
        ipywidgets.VBox
        """
        sections_to_include = html.verify_include_parameter(check_sections)
        sections: t.List[Widget] = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(self.prepare_conditions_table(output_id=output_id))

        if 'additional-output' in sections_to_include:
            sections.append(self.prepare_additional_output(output_id=output_id))

        return normalize_widget_style(VBox(children=sections))

    def prepare_header(self, output_id: t.Optional[str] = None) -> HTML:
        """Prepare header widget."""
        return HTML(value=self._html_serializer.prepare_header(output_id))

    def prepare_summary(self) -> HTML:
        """Prepare summary widget."""
        return HTML(value=self._html_serializer.prepare_summary())

    def prepare_conditions_table(
        self,
        max_info_len: int = 3000,
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> HTML:
        """Prepare conditions table widget.

        Parameters
        ----------
        max_info_len : int, default 3000
            max length of the additional info
        include_icon : bool , default: True
            if to show the html condition result icon or the enum
        include_check_name : bool, default False
            whether to include check name into dataframe or not
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.HTML
        """
        widget = HTML(value=self._html_serializer.prepare_conditions_table(
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        ))
        return widget

    def prepare_additional_output(self, output_id: t.Optional[str] = None) -> VBox:
        """Prepare additional output widget.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.VBox
        """
        return VBox(children=DisplayItemsHandler.handle_display(
            self.value.display,
            output_id,
        ))


class DisplayItemsHandler(html.DisplayItemsHandler):
    """Auxiliary class to decouple display handling logic from other functionality."""

    @classmethod
    def header(cls) -> HTML:
        """Return header section."""
        return HTML(value=super().header())

    @classmethod
    def empty_content_placeholder(cls) -> HTML:
        """Return placeholder in case of content absence."""
        return HTML(value=super().empty_content_placeholder())

    @classmethod
    def go_to_top_link(cls, output_id: str) -> HTML:
        """Return 'Go To Top' link."""
        return HTML(value=super().go_to_top_link(output_id))

    @classmethod
    def handle_figure(cls, item: BaseFigure, index: int, **kwargs) -> Widget:  # pylint: disable=unused-argument
        return go.FigureWidget(data=item)

    @classmethod
    def handle_string(cls, item: str, index: int, **kwargs) -> HTML:
        """Handle textual item."""
        return HTML(value=super().handle_string(item, index, **kwargs))

    @classmethod
    def handle_dataframe(cls, item: pd.DataFrame, index: int, **kwargs) -> HTML:
        """Handle dataframe item."""
        return HTML(value=super().handle_dataframe(item, index, **kwargs))

    @classmethod
    def handle_callable(cls, item: t.Callable, index: int, **kwargs) -> HTML:
        """Handle callable."""
        return HTML(value=super().handle_callable(item, index, **kwargs))
