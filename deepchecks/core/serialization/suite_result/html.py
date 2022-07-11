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
"""Module containing html serializer for the SuiteResult type."""
import textwrap
import typing as t
import warnings

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.resources import requirejs_script
from deepchecks.core.serialization.abc import HtmlSerializer
from deepchecks.core.serialization.check_result.html import CheckResultSection
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as CheckResultHtmlSerializer
from deepchecks.core.serialization.common import (Html, aggregate_conditions, create_failures_dataframe,
                                                  form_output_anchor, plotlyjs_script)
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer as DataFrameHtmlSerializer
from deepchecks.utils.html import linktag

__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(HtmlSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into HTML format.

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

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        full_html: bool = False,
        include_requirejs: bool = False,
        include_plotlyjs: bool = True,
        connected: bool = True,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs,
    ) -> str:
        """Serialize a SuiteResult instance into HTML format.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        full_html : bool, default False
            whether to return a fully independent HTML document or only CheckResult content
        include_requirejs : bool, default False
            whether to include requirejs library into output or not
        include_plotlyjs : bool, default True
            whether to include plotlyjs library into output or not
        connected : bool, default True
            whether to use CDN to load js libraries or to inject their code into output
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        str
        """
        if full_html is True:
            include_plotlyjs = True
            include_requirejs = True
            connected = False

        kwargs['is_for_iframe_with_srcdoc'] = is_for_iframe_with_srcdoc

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
            Html.bold_hr,
            conditions_table,
            Html.bold_hr,
            results_with_conditions,
            Html.bold_hr,
            results_without_conditions,
        ]

        if failures:
            sections.extend([Html.bold_hr, failures])

        if output_id:
            anchor = form_output_anchor(output_id)
            link = linktag(
                text='Go to top',
                href=f'#{anchor}',
                style={'font-size': '14px'},
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )
            sections.append(f'<br>{link}')

        plotlyjs = plotlyjs_script(connected) if include_plotlyjs is True else ''
        requirejs = requirejs_script(connected) if include_requirejs is True else ''

        if full_html is False:
            return ''.join([requirejs, plotlyjs, *sections])

        # TODO: use some style to make it pretty
        return textwrap.dedent(f"""
            <html>
            <head><meta charset="utf-8"/></head>
            <body style="background-color: white; padding: 1rem 1rem 0 1rem;">
                {requirejs}
                {plotlyjs}
                {''.join(sections)}
            </body>
            </html>
        """)

    def prepare_prologue(self) -> str:
        """Prepare prologue section."""
        long_prologue_version = 'The suite is composed of various checks such as: {names}, etc...'
        short_prologue_version = 'The suite is composed of the following checks: {names}.'
        check_names = list(set(
            it.check.name()
            for it in self.value.results
        ))
        return (
            long_prologue_version.format(names=', '.join(check_names[:3]))
            if len(check_names) > 3
            else short_prologue_version.format(names=', '.join(check_names))
        )

    def prepare_header(self, output_id: t.Optional[str] = None, **kwargs) -> str:
        """Prepare header section.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        str
        """
        idattr = f' id="{form_output_anchor(output_id)}"' if output_id else ''
        return f'<h1{idattr}>{self.value.name}</h1>'

    def prepare_extra_info(self) -> str:
        """Prepare extra info section."""
        if self.value.extra_info:
            extra_info = '<br>'.join(f'<div>{it}</div>' for it in self.value.extra_info)
            return f'<br>{extra_info}'
        else:
            return ''

    def prepare_summary(self, output_id: t.Optional[str] = None, **kwargs) -> str:
        """Prepare summary section.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        str
        """
        header = self.prepare_header(output_id)
        prologue = self.prepare_prologue()
        extra_info = self.prepare_extra_info()

        suite_creation_example_link = (
            'https://docs.deepchecks.com/en/stable/examples/guides/create_a_custom_suite.html'
            '?utm_source=display_output&utm_medium=referral&utm_campaign=suite_link'
        )
        icons = textwrap.dedent("""
            <span style="color: green;display:inline-block">\U00002713</span> /
            <span style="color: red;display:inline-block">\U00002716</span> /
            <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span> /
            <span style="color: firebrick;font-weight:bold;display:inline-block">\U00002048</span>
        """)
        return textwrap.dedent(f"""
            {header}
            <p>
                {prologue}<br>
                Each check may contain conditions (which will result in pass / fail / warning / error
                , represented by {icons}) as well as other outputs such as plots or tables.<br>
                Suites, checks and conditions can all be modified. Read more about
                <a href={suite_creation_example_link} target="_blank">custom suites</a>.
            </p>
            {extra_info}
        """)

    def prepare_conditions_table(
        self,
        output_id: t.Optional[str] = None,
        include_check_name: bool = True,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs
    ) -> str:
        """Prepare conditions table section.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        include_check_name : bool, default True
            wherether to include check name into table or not
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        str
        """
        if not self.value.results_with_conditions:
            return '<p>No conditions defined on checks in the suite.</p>'

        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(self.value.results_with_conditions)
        )
        table = DataFrameHtmlSerializer(aggregate_conditions(
            results,
            output_id=output_id,
            include_check_name=include_check_name,
            max_info_len=300,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        )).serialize()

        return f'<h2>Conditions Summary</h2>{table}'

    def prepare_results_with_condition_and_display(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> str:
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
        str
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(
                self.value.results_with_conditions & self.value.results_with_display
            )
        )
        results_with_condition_and_display = [
            CheckResultHtmlSerializer(it).serialize(
                output_id=output_id,
                check_sections=check_sections,
                include_plotlyjs=False,
                include_requirejs=False,
                **kwargs
            )
            for it in results
        ]
        content = Html.light_hr.join(results_with_condition_and_display)
        return f'<h2>Check With Conditions Output</h2>{content}'

    def prepare_results_without_condition(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        **kwargs
    ) -> str:
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
        str
        """
        results = t.cast(
            t.List[check_types.CheckResult],
            self.value.select_results(
                self.value.results_without_conditions & self.value.results_with_display,
            )
        )
        results_without_conditions = [
            CheckResultHtmlSerializer(it).serialize(
                output_id=output_id,
                include=check_sections,
                include_plotlyjs=False,
                include_requirejs=False,
                **kwargs
            )
            for it in results
        ]
        content = Html.light_hr.join(results_without_conditions)
        return f'<h2>Check Without Conditions Output</h2>{content}'

    def prepare_failures_list(self, **kwargs) -> str:
        """Prepare subsection of the content that shows list of failures."""
        results = self.value.select_results(self.value.failures | self.value.results_without_display)

        if not results:
            return ''

        df = create_failures_dataframe(results)

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            table = DataFrameHtmlSerializer(df.style.hide_index()).serialize()
            return f'<h2>Other Checks That Weren\'t Displayed</h2>\n{table}'
