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
"""Module containing SuiteResult serialization logic."""
import typing as t

from IPython.display import HTML

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import IPythonFormatter, IPythonSerializer
from deepchecks.core.serialization.check_result.html import CheckResultSection
from deepchecks.core.serialization.check_result.ipython import CheckResultSerializer
from deepchecks.core.serialization.common import Html, flatten, form_output_anchor, join
from deepchecks.utils.html import linktag

from . import html

__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(IPythonSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into a list of IPython formatters.

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
        super().__init__(value=value)
        self._html_serializer = html.SuiteResultSerializer(value)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs,
    ) -> t.List[IPythonFormatter]:
        """Serialize a SuiteResult instance into a list of IPython formatters.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor link or not
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        List[IPythonFormatter]
        """
        summary = self.prepare_summary(output_id=output_id, **kwargs)
        conditions_table = self.prepare_conditions_table(output_id=output_id, **kwargs)
        failures = self.prepare_failures_list()

        results_with_conditions = self.prepare_results_with_condition_and_display(
            output_id=output_id,
            check_sections=['condition-table', 'additional-output'],
            **kwargs
        )
        results_without_conditions = self.prepare_results_without_condition(
            output_id=output_id,
            check_sections=['additional-output'],
            **kwargs
        )
        sections = [
            summary,
            HTML(Html.bold_hr),
            conditions_table,
            HTML(Html.bold_hr),
            results_with_conditions,
            HTML(Html.bold_hr),
            results_without_conditions,
        ]

        if failures:
            sections.extend([HTML(Html.bold_hr), failures])

        if output_id:
            link = linktag(
                text='Go to top',
                href=f'#{form_output_anchor(output_id)}',
                style={'font-size': '14px'},
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )
            sections.append(HTML(f'<br>{link}'))

        return list(flatten(sections))

    def prepare_summary(self, output_id: t.Optional[str] = None, **kwargs) -> HTML:
        """Prepare summary section.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        HTML
        """
        return HTML(self._html_serializer.prepare_summary(
            output_id=output_id,
            **kwargs
        ))

    def prepare_conditions_table(
        self,
        output_id: t.Optional[str] = None,
        include_check_name: bool = True,
        **kwargs
    ) -> HTML:
        """Prepare conditions table section.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        include_check_name : bool, default True
            wherether to include check name into table or not

        Returns
        -------
        HTML
        """
        return HTML(self._html_serializer.prepare_conditions_table(
            output_id=output_id,
            include_check_name=include_check_name,
            **kwargs
        ))

    def prepare_results_with_condition_and_display(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> t.List[IPythonFormatter]:
        """Prepare subsection of the content that shows results with conditions.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included

        Returns
        -------
        List[IPythonFormatter]
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(
                self.value.results_with_conditions & self.value.results_with_display
            )
        )
        results_with_condition_and_display = [
            CheckResultSerializer(it).serialize(
                output_id=output_id,
                check_sections=check_sections,
                **kwargs
            )
            for it in results
        ]
        content = join(
            results_with_condition_and_display,
            HTML(Html.light_hr)
        )
        return list(flatten([
            HTML('<h2>Check With Conditions Output</h2>'),
            content
        ]))

    def prepare_results_without_condition(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> t.List[IPythonFormatter]:
        """Prepare subsection of the content that shows results without conditions.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included

        Returns
        -------
        List[IPythonFormatter]
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(
                self.value.results_without_conditions & self.value.results_with_display,
            )
        )
        results_without_conditions = [
            CheckResultSerializer(it).serialize(
                output_id=output_id,
                include=check_sections,
                include_plotlyjs=False,
                include_requirejs=False,
                **kwargs
            )
            for it in results
        ]
        content = join(
            results_without_conditions,
            HTML(Html.light_hr)
        )
        return list(flatten([
            HTML('<h2>Check Without Conditions Output</h2>'),
            content
        ]))

    def prepare_failures_list(self) -> HTML:
        """Prepare subsection of the content that shows list of failures."""
        return HTML(self._html_serializer.prepare_failures_list())
