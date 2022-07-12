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

from plotly.offline.offline import get_plotlyjs

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.resources import DEEPCHECKS_HTML_PAGE_STYLE, DEEPCHECKS_STYLE
from deepchecks.core.serialization.abc import HtmlSerializer
from deepchecks.core.serialization.check_failure.html import CheckFailureSerializer as CheckFailureHtmlSerializer
from deepchecks.core.serialization.check_result.html import CheckEmbedmentWay
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as CheckResultHtmlSerializer
from deepchecks.core.serialization.common import STYLE_LOADER, PLOTLY_LOADER
from deepchecks.core.serialization.common import (Html, aggregate_conditions,
                                                  create_failures_dataframe, create_results_dataframe,
                                                  form_output_anchor, contains_plots)
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer
from deepchecks.utils.html import details_tag, iframe_tag
from deepchecks.utils.strings import get_random_string

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
        collapsible: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
        embed_into_iframe: bool = False,
        use_javascript: bool = True,
        **kwargs,
    ) -> str:
        """Serialize a SuiteResult instance into HTML format.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        full_html : bool, default False
            whether to return a fully independent HTML document or not
        collapsible : bool, default False
            Note: this parameter is used only when the 'full_html' parameter is
            set to 'True' otherwise it is totally ignored.
            It tells a serializer whether the suite result output must be
            embedded into HTML '<details>' tag or not.
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not
        embed_into_iframe : bool , default False
            whether to embed output into iframe or not
        use_javascript : bool , default True
            whether to use  javascript in an output or not.
            If set to 'False', all components that require javascript
            to work will be replaced by plain HTML components (if possible).
            For example, plotly figures will be transformed into JPEG images,
            tabs widgets will be replaced by HTML '<details>' tag
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        str
        """
        if embed_into_iframe is True:
            is_for_iframe_with_srcdoc = True

        check_embedment_way = 'suite-html-page' if full_html or embed_into_iframe else 'suite'
        kwargs['is_for_iframe_with_srcdoc'] = is_for_iframe_with_srcdoc
        kwargs['use_javascript'] = use_javascript

        suite_result = self.value
        passed_checks = suite_result.get_passed_checks()
        not_passed_checks = suite_result.get_not_passed_checks()
        not_ran_checks = suite_result.get_not_ran_checks()
        other_checks = t.cast(
            t.List[check_types.CheckResult],
            suite_result.select_results(self.value.results_without_conditions)
        )

        content = (
            self.prepare_summary(
                output_id=output_id,
                **kwargs
            ),
            self.prepare_results(
                title='Didn`t Pass',
                results=not_passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                check_embedment_way=check_embedment_way,
                **kwargs
            ),
            self.prepare_results(
                title='Passed',
                results=passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                check_embedment_way=check_embedment_way,
                **kwargs
            ),
            self.prepare_results(
                title='Other',
                results=other_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_unconditioned_results_summary,
                check_sections=['additional-output'],
                check_embedment_way=check_embedment_way,
                **kwargs
            ),
            self.prepare_failures(
                title='Didn`t Run',
                failures=not_ran_checks,
                output_id=output_id,
                **kwargs
            )
        )

        if embed_into_iframe is True:
            return self._serialize_to_iframe(content, use_javascript)

        if full_html is True:
            return self._serialize_to_full_html(content, collapsible, use_javascript)

        output = details_tag(
            title=suite_result.name,
            content=''.join(content),
            id=output_id or '',
            attrs='open class="deepchecks"',
        )

        if use_javascript is False:
            return f'<style>{DEEPCHECKS_STYLE}</style>{output}'
        
        if not contains_plots(self.value):
            return f'{STYLE_LOADER}{output}'

        return f'{STYLE_LOADER}{PLOTLY_LOADER}{output}'

    def _serialize_to_full_html(
        self,
        content: t.Sequence[str],
        collapsible: bool = False,
        use_javascript: bool = True
    ) -> str:
        content = (
            details_tag(
                title=self.value.name,
                content=''.join(content),
                attrs='open class="deepchecks"',
            )
            if collapsible is True
            else ''.join(content)
        )
        script = (
            f'<script type="text/javascript">{get_plotlyjs()}</script>'
            if use_javascript
            else ''
        )
        return f"""
            <html>
            <head>
                <meta charset="utf-8"/>
                <style>{DEEPCHECKS_HTML_PAGE_STYLE}</style>
                {script}
            </head>
            <body class="deepchecks">{content}</body>
            </html>
        """

    def _serialize_to_iframe(
        self,
        content: t.Sequence[str],
        use_javascript: bool = True
    ) -> str:
        content = iframe_tag(
            title=self.value.name,
            collapsible=True,
            srcdoc=self._serialize_to_full_html(
                content,
                collapsible=False,
                use_javascript=use_javascript
            )
        )
        return f'<style>{DEEPCHECKS_STYLE}</style>{content}'

    def prepare_prologue(self) -> str:
        """Prepare prologue section."""
        long_prologue_version = 'The suite is composed of various checks such as: {names}, etc...'
        short_prologue_version = 'The suite is composed of the following checks: {names}.'
        check_names = set(it.check.name() for it in self.value.results)
        return (
            long_prologue_version.format(names=', '.join(tuple(check_names)[:3]))
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
        return f'<h1{idattr}><b>{self.value.name}</b></h1>'

    def prepare_extra_info(self) -> str:
        """Prepare extra info section."""
        if self.value.extra_info:
            return ''.join(f'<p>{it}</p>' for it in self.value.extra_info)
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
            </p>
            <p>
                Suites, checks and conditions can all be modified. Read more about
                <a href={suite_creation_example_link} target="_blank">custom suites</a>.
            </p>
            {extra_info}
        """)

    def prepare_conditions_summary(
        self,
        results: t.Sequence['check_types.CheckResult'],
        output_id: t.Optional[str] = None,
        include_check_name: bool = True,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs
    ) -> str:
        """Prepare conditions summary table.

        Parameters
        ----------
        results : Sequence[CheckResult]
            sequence of check results
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
        return DataFrameSerializer(aggregate_conditions(
            results,
            output_id=output_id,
            include_check_name=include_check_name,
            max_info_len=300,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        )).serialize()

    def prepare_unconditioned_results_summary(
        self,
        results: t.Sequence['check_types.CheckResult'],
        output_id: t.Optional[str] = None,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs
    ) -> str:
        """Prepare results summary table.

        Parameters
        ----------
        results : Sequence[CheckResult]
            sequence of check results
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        str
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df = create_results_dataframe(
                results=results,
                output_id=output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )
            return DataFrameSerializer(df.style.hide_index()).serialize()

    def prepare_failures(
        self,
        failures: t.Sequence['check_types.CheckFailure'],
        title: str,
        **kwargs
    ) -> str:
        """Prepare failures section.

        Parameters
        ----------
        failures : Sequence[CheckFailure]
            sequence of check failures
        title : str
            accordion title

        Returns
        -------
        str
        """
        if len(failures) == 0:
            content = '<p>No outputs to show.</p>'
        else:
            df = create_failures_dataframe(failures)
            content = DataFrameSerializer(df.style.hide_index()).serialize()

        return details_tag(
            title=title,
            content=content,
            attrs='class="deepchecks"'
        )

    def prepare_results(
        self,
        results: t.Sequence['check_types.CheckResult'],
        title: str,
        output_id: t.Optional[str] = None,
        summary_creation_method: t.Optional[t.Callable[..., str]] = None,
        check_embedment_way: CheckEmbedmentWay = 'suite',
        **kwargs
    ) -> str:
        """Prepare results section.

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
        check_embedment_way : Union[Literal['suite'], Literal['suite-html-page']], default 'suite'
            check embedment strategy, it tells the 'CheckResult' serializer
            in what way Plotly figures must be serialized into HTML.

        Returns
        -------
        str
        """
        if len(results) == 0:
            section_id = None
            content = '<p>No outputs to show.</p>'
        else:
            section_id = (
                f'{output_id}-section-{get_random_string()}'
                if output_id is not None
                else None
            )
            serialized_results = (
                select_serializer(it).serialize(
                    output_id=section_id,
                    embed_into_suite=check_embedment_way,
                    **kwargs
                )
                for it in results
                if it.display  # we do not form full-output for the check results without display
            )
            if callable(summary_creation_method):
                content = Html.light_hr.join((
                    summary_creation_method(results=results, output_id=section_id, **kwargs),
                    *serialized_results
                ))
            else:
                content = Html.light_hr.join(serialized_results)

        return details_tag(
            title=title,
            content=content,
            id=form_output_anchor(section_id) if section_id else None,
            attrs='class="deepchecks"',
        )


def select_serializer(result):
    if isinstance(result, check_types.CheckResult):
        return CheckResultHtmlSerializer(result)
    elif isinstance(result, check_types.CheckFailure):
        return CheckFailureHtmlSerializer(result)
    else:
        raise TypeError(f'Unknown type of result - {type(result)}')
