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
from plotly.basedatatypes import BaseFigure
from ipywidgets import HTML, VBox, Widget

from deepchecks.core.check_result import CheckResult
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.common import normalize_widget_style
from . import html


__all__ = ['CheckResultSerializer']


class CheckResultSerializer(WidgetSerializer[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        super().__init__(**{'value': value, **kwargs})
        self.value = value
        self._html_serializer = html.CheckResultSerializer(self.value)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        include: t.Optional[t.Sequence[html.CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        sections_to_include = html.verify_include_parameter(include)
        sections: t.List[Widget] = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(self.prepare_conditions_table(output_id=output_id))

        if 'additional-output' in sections_to_include:
            sections.append(self.prepare_additional_output(output_id))

        return normalize_widget_style(VBox(children=sections))

    def prepare_header(self, output_id: t.Optional[str] = None) -> HTML:
        return HTML(value=self._html_serializer.prepare_header(output_id))

    def prepare_summary(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_summary())

    def prepare_conditions_table(
        self,
        max_info_len: int = 3000,
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> HTML:
        widget = HTML(value=self._html_serializer.prepare_conditions_table(
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        ))
        return widget

    def prepare_additional_output(self, output_id: t.Optional[str] = None) -> VBox:
        return VBox(children=DisplayItemsHandler.handle_display(
            self.value.display,
            output_id
        ))


class DisplayItemsHandler(html.DisplayItemsHandler):

    @classmethod
    def header(cls) -> HTML:
        return HTML(value=super().header())

    @classmethod
    def empty_content_placeholder(cls) -> HTML:
        return HTML(value=super().empty_content_placeholder())

    @classmethod
    def go_to_top_link(cls, output_id: str) -> HTML:
        return HTML(value=super().go_to_top_link(output_id))

    @classmethod
    def handle_figure(cls, item: BaseFigure) -> go.FigureWidget:
        return go.FigureWidget(data=item)

    @classmethod
    def handle_string(cls, item: str) -> HTML:
        return HTML(value=super().handle_string(item))

    @classmethod
    def handle_dataframe(cls, item: pd.DataFrame) -> HTML:
        return HTML(value=super().handle_dataframe(item))

    @classmethod
    def handle_callable(cls, item: t.Callable) -> HTML:
        raise NotImplementedError()
