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
from ipywidgets import HTML, Tab, VBox, Widget
from plotly.basedatatypes import BaseFigure

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import DisplayItems
from deepchecks.core.serialization.abc import DisplayItemsSerializer as ABCDisplayItemsSerializer
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.common import Html as CommonHtml
from deepchecks.core.serialization.common import (figure_to_html_image_tag, go_to_top_link, normalize_widget_style,
                                                  read_matplot_figures_as_html)
from deepchecks.core.serialization.dataframe.widget import DataFrameSerializer

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
        super().__init__(value=value)
        self._html_serializer = html.CheckResultSerializer(self.value)
        self._display_serializer = DisplayItemsSerializer(self.value.display)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[html.CheckResultSection]] = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
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
        plotly_to_image : bool, default False
            whether to transform Plotly figure instance into static image or not
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        ipywidgets.VBox
        """
        sections_to_include = html.verify_include_parameter(check_sections)
        sections: t.List[Widget] = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(self.prepare_conditions_table(
                output_id=output_id
            ))

        if 'additional-output' in sections_to_include:
            sections.append(self.prepare_additional_output(
                output_id=output_id,
                plotly_to_image=plotly_to_image,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

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
        return HTML(value=self._html_serializer.prepare_conditions_table(
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        ))

    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False
    ) -> VBox:
        """Prepare additional output widget.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        plotly_to_image : bool, default False
            whether to transform Plotly figure instance into static image or not
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        ipywidgets.VBox
        """
        output: t.List[Widget] = [HTML(value=CommonHtml.additional_output_header)]

        if len(self.value.display) == 0:
            output.append(HTML(value=CommonHtml.empty_content_placeholder))
        else:
            output.extend(self._display_serializer.serialize(
                output_id=output_id,
                plotly_to_image=plotly_to_image,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

        if output_id is not None:
            output.append(HTML(value=go_to_top_link(
                output_id=output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )))

        return VBox(children=output)


class DisplayItemsSerializer(ABCDisplayItemsSerializer[Widget]):
    """CheckResult display items serializer."""

    def __init__(self, value: DisplayItems, **kwargs):
        super().__init__(value, **kwargs)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> t.List[Widget]:
        """Serialize CheckResult display items into list if Widget instances.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        List[Widget]
        """
        return self.handle_display(self.value, output_id=output_id, **kwargs)

    def handle_figure(
        self,
        item: BaseFigure,
        index: int,
        plotly_to_image: bool = False,
        **kwargs
    ) -> Widget:
        """Handle plotly figure."""
        return (
            go.FigureWidget(data=item)
            if not plotly_to_image
            else HTML(value=figure_to_html_image_tag(item))
        )

    def handle_string(self, item: str, index: int, **kwargs) -> HTML:
        """Handle textual item."""
        return HTML(value=item)

    def handle_dataframe(self, item: pd.DataFrame, index: int, **kwargs) -> HTML:
        """Handle dataframe item."""
        return DataFrameSerializer(item).serialize()

    def handle_callable(self, item: t.Callable, index: int, **kwargs) -> HTML:
        """Handle callable."""
        return HTML(value=''.join(read_matplot_figures_as_html(item)))

    def handle_display_map(self, item: 'check_types.DisplayMap', index: int, **kwargs) -> VBox:
        """Handle display map instance item."""
        tab = Tab()
        children = []

        for i, (name, display) in enumerate(item.items()):
            tab.set_title(i, name)
            children.append(VBox(children=self.handle_display(
                display,
                include_header=False,
                include_trailing_link=False,
                **kwargs
            )))

        tab.children = children
        style = '<style>.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {min-width: fit-content;}</style>'
        return VBox(children=[HTML(value=style), tab])
