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

from ipywidgets import HTML, VBox, Widget, Accordion

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.core.serialization.check_failure.widget import CheckFailureSerializer as CheckFailureWidgetSerializer
from deepchecks.core.serialization.common import Html as CommonHtml
from deepchecks.core.serialization.common import join, normalize_widget_style, aggregate_conditions, create_results_dataframe, form_output_anchor, create_failures_dataframe
from deepchecks.core.serialization.dataframe.widget import DataFrameSerializer
from deepchecks.utils.strings import get_random_string

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
    ) -> Widget:
        """Serialize a SuiteResult instance into ipywidgets.Widget instance.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.VBox
        """
        passed_checks = self.value.get_passed_checks()
        not_passed_checks = self.value.get_not_passed_checks()
        not_ran_checks = self.value.get_not_ran_checks()
        other_checks = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(self.value.results_without_conditions)
        )
        
        accordions = [
            self.prepare_results(
                title='Passed',
                results=passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                **kwargs
            ),
            self.prepare_results(
                title='Not Passed',
                results=not_passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                **kwargs
            ),
            self.prepare_failures(
                title='Did not run',
                failures=not_ran_checks,
                output_id=output_id,
                **kwargs
            ),
            self.prepare_results(
                title='Other',
                results=other_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_unconditioned_results_summary,
                check_sections=['additional-output'],
                **kwargs
            )
        ]

        style = '<style>.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {flex: 0 1 auto}</style>'

        content = VBox(children=[
            HTML(value=style),
            self.prepare_summary(output_id=output_id, **kwargs),
            *accordions
        ])
        return Accordion(
            children=[content], 
            _titles={'0': self.value.name},
            selected_index='0'
        )

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

    def prepare_failures(
        self,
        failures: t.Sequence['check_types.CheckFailure'],
        title: str,
        **kwargs
    ) -> Accordion:
        """Prepare accordion with failures.

        Parameters
        ----------
        failures : Sequence[CheckFailure]
            sequence of check failures
        title : str
            accordion title

        Returns
        -------
        ipywidgets.Accordion
        """
        if len(failures) == 0:
            children = (HTML(value='<p>No outputs to show.</p>'),)
        else:
            df = create_failures_dataframe(failures)
            table = DataFrameSerializer(df.style.hide_index()).serialize()
            children = (table,)
        return Accordion(
            children=children, 
            _titles={'0': title},
            selected_index=None
        )
    
    def prepare_results(
        self,
        results: t.Sequence['check_types.CheckResult'],
        title: str,
        output_id: t.Optional[str] = None,
        summary_creation_method: t.Optional[t.Callable[..., Widget]] = None,
        **kwargs
    ) -> Accordion:
        """Prepare accordion with results.

        Parameters
        ----------
        results : Sequence[CheckResult]
            sequence of check results
        title : str
            accordion title
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        summary_creation_method : Optional[Callable[..., Widget]], default None
            function to create summary table 

        Returns
        -------
        ipywidgets.Accordion
        """
        if len(results) == 0:
            children = (HTML(value='<p>No outputs to show.</p>'),)
        else:
            section_id = f'{output_id}-section-{get_random_string()}'
            serialized_results = [
                select_serializer(it).serialize(output_id=section_id, **kwargs)
                for it in results
                if it.display  # we do not form full-output for the check results without display
            ]
            if callable(summary_creation_method):
                children = (
                    HTML(value=f'<span id="{form_output_anchor(section_id)}"></span>'),
                    summary_creation_method(results=results, output_id=section_id),
                    HTML(value=CommonHtml.light_hr),
                    *join(serialized_results, HTML(value=CommonHtml.light_hr))
                )
            else:
                children = (
                    HTML(value=f'<span id="{form_output_anchor(section_id)}"></span>'),
                    *join(serialized_results, HTML(value=CommonHtml.light_hr)),
                )
        return normalize_widget_style(Accordion(
            children=(VBox(children=children),),
            _titles={'0': title},
            selected_index=None
        ))
    
    def prepare_conditions_summary(
        self,
        results: t.Sequence['check_types.CheckResult'],
        output_id: t.Optional[str] = None,
        include_check_name: bool = True,
        **kwargs
    ) -> Widget:
        """Prepare conditions summary table.

        Parameters
        ----------
        results : Sequence[CheckResult]
            sequence of check results
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        include_check_name : bool, default True
            wherether to include check name into table or not

        Returns
        -------
        ipywidgets.Widget
        """
        table = DataFrameSerializer(aggregate_conditions(
            results,
            output_id=output_id,
            include_check_name=include_check_name,
            max_info_len=300
        )).serialize()
        return table

    def prepare_unconditioned_results_summary(
        self,
        results: t.Sequence['check_types.CheckResult'],
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> Widget:
        """Prepare results summary table.

        Parameters
        ----------
        results : Sequence[CheckResult]
            sequence of check results
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        ipywidgets.Widget
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df = create_results_dataframe(results=results, output_id=output_id)
            return DataFrameSerializer(df.style.hide_index()).serialize()


def select_serializer(result):
    if isinstance(result, check_types.CheckResult):
        return CheckResultWidgetSerializer(result)
    elif isinstance(result, check_types.CheckFailure):
        return CheckFailureWidgetSerializer(result)
    else:
        raise TypeError(f'Unknown type of result - {type(result)}') 
