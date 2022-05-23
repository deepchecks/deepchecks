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
"""Module containing ipywidget serializer for the SuiteResult type."""
import typing as t
import warnings

import pandas as pd
from ipywidgets import HTML, Tab, VBox, Widget

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.check_result.html import CheckResultSection
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.core.serialization.common import Html as CommonHtml
from deepchecks.core.serialization.common import join, normalize_widget_style
from deepchecks.core.serialization.dataframe.widget import DataFrameSerializer

from . import html

__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(WidgetSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into ipywidgets.Widget instance.

    Parameters
    ----------
    value : SuiteResult
        SuiteResult instance that needed to be serialized.
    """

    def __init__(self, value: 'suite.SuiteResult', **kwargs):
        if not isinstance(value, suite.SuiteResult):
            raise TypeError(
                f'Expected "SuiteResult" but got "{type(value).__name__}"'
            )
        self.value = value
        self._html_serializer = html.SuiteResultSerializer(self.value)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> VBox:
        """Serialize a SuiteResult instance into ipywidgets.Widget instance.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.VBox
        """
        tab = Tab()
        tab.set_title(0, 'Checks With Conditions')
        tab.set_title(1, 'Checks Without Conditions')
        tab.set_title(2, 'Checks Without Output')

        tab.children = [
            self.prepare_results_with_condition_and_display(
                output_id=output_id, **kwargs
            ),
            self.prepare_results_without_condition(
                output_id=output_id,
                check_sections=['additional-output'],
                **kwargs
            ),
            self.prepare_failures_list()
        ]

        style = '<style>.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {flex: 0 1 auto}</style>'

        return VBox(children=[
            HTML(value=style),
            self.prepare_summary(output_id=output_id, **kwargs),
            tab
        ])

    def prepare_summary(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> HTML:
        """Prepare summary widget."""
        return HTML(value=self._html_serializer.prepare_summary(
            output_id=output_id,
            **kwargs
        ))

    def prepare_conditions_table(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> HTML:
        """Prepare summary widget."""
        return normalize_widget_style(HTML(value=self._html_serializer.prepare_conditions_table(
            output_id=output_id,
            include_check_name=True,
            **kwargs
        )))

    def prepare_failures_list(self) -> HTML:
        """Prepare failures list widget."""
        return normalize_widget_style(HTML(
            value=self._html_serializer.prepare_failures_list() or '<p>No outputs to show.</p>'
        ))

    def prepare_results_without_condition(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        """Prepare widget that shows results without conditions.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included

        Returns
        -------
        ipywidgets.VBox
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(self.value.results_without_conditions & self.value.results_with_display)
        )
        results_without_conditions = [
            CheckResultWidgetSerializer(it).serialize(
                output_id=output_id,
                include=check_sections,
                **kwargs
            )
            for it in results
        ]
        if len(results_without_conditions) > 0:
            children = (
                HTML(value='<h2>Check Without Conditions Output</h2>'),
                self.prepare_navigation_for_unconditioned_results(output_id),
                HTML(value=CommonHtml.light_hr),
                *join(results_without_conditions, HTML(value=CommonHtml.light_hr))
            )
        else:
            children = (
                HTML(value='<p>No outputs to show.</p>'),
            )

        return normalize_widget_style(VBox(children=children))

    def prepare_results_with_condition_and_display(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> VBox:
        """Prepare widget that shows results with conditions and display.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included

        Returns
        -------
        ipywidgets.VBox
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(self.value.results_with_conditions & self.value.results_with_display)
        )
        results_with_condition_and_display = [
            CheckResultWidgetSerializer(it).serialize(
                output_id=output_id,
                include=check_sections,
                **kwargs
            )
            for it in results
        ]

        if len(results_with_condition_and_display) > 0:
            children = (
                self.prepare_conditions_table(output_id=output_id),
                HTML(value='<h2>Check With Conditions Output</h2>'),
                *join(results_with_condition_and_display, HTML(value=CommonHtml.light_hr))
            )
        else:
            children = (
                HTML(value='<p>No outputs to show.</p>'),
            )
        return normalize_widget_style(VBox(children=children))

    def prepare_navigation_for_unconditioned_results(
        self,
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> Widget:
        """Prepare navigation widget for the tab with unconditioned_results.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.Widget
        """
        data = []

        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(self.value.results_without_conditions & self.value.results_with_display)
        )

        for check_result in results:
            check_header = check_result.get_header()

            if output_id:
                href = f'href="#{check_result.get_check_id(output_id)}"'
                header = f'<a {href}>{check_header}</a>'
            else:
                header = check_header

            summary = check_result.get_metadata(with_doc_link=True)['summary']
            data.append([header, summary])

        df = pd.DataFrame(
            data=data,
            columns=['Check', 'Summary']
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            return DataFrameSerializer(df.style.hide_index()).serialize()
