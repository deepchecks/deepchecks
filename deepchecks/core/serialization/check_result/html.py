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
"""Module containing html serializer for the CheckResult type."""
import typing as t

from plotly.basedatatypes import BaseFigure
from plotly.io import to_html
from plotly.offline.offline import get_plotlyjs
from typing_extensions import Literal as L

from deepchecks.core import check_result as check_types
from deepchecks.core.resources import DEEPCHECKS_HTML_PAGE_STYLE, DEEPCHECKS_STYLE
from deepchecks.core.serialization.abc import DisplayItemsSerializer as ABCDisplayItemsSerializer
from deepchecks.core.serialization.abc import HtmlSerializer
from deepchecks.core.serialization.common import PLOTLY_LOADER, STYLE_LOADER
from deepchecks.core.serialization.common import Html as CommonHtml
from deepchecks.core.serialization.common import (aggregate_conditions, contains_plots, figure_to_html,
                                                  figure_to_html_image_tag, go_to_top_link,
                                                  read_matplot_figures_as_html)
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer as DataFrameHtmlSerializer
from deepchecks.utils.html import details_tag, iframe_tag, tabs_widget

__all__ = ['CheckResultSerializer']


CheckResultSection = t.Union[
    L['condition-table'],  # noqa
    L['additional-output'],  # noqa
]


CheckEmbedmentWay = t.Union[
    L['suite'],            # embedded into suite display  # noqa
    L['suite-html-page'],  # embedded into suite display serialized with 'full_html' flag set to True  # noqa
]


class CheckResultSerializer(HtmlSerializer['check_types.CheckResult']):
    """Serializes any CheckResult instance into HTML format.

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
        self._display_serializer = DisplayItemsSerializer(self.value.display)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        embed_into_iframe: bool = False,
        full_html: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
        use_javascript: bool = True,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        **kwargs
    ) -> str:
        """Serialize a CheckResult instance into HTML format.

        Parameters
        ----------
        output_id : Optional[str] , default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]] , default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included
        embed_into_iframe : bool , default False
            whether to embed output into iframe or not
        full_html : bool , default False
            whether to return a fully independent HTML document or not.
            NOTE: this parameter is ignored if the 'embed_into_iframe' parameter
            is set to 'True'.
        is_for_iframe_with_srcdoc : bool , default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not.
            NOTE: Is automatically set to 'True' if the 'embed_into_iframe' parameter is
            set to 'True'
        use_javascript : bool , default True
            whether to use javascript in an output or not.
            If set to 'False', all components that require javascript
            to work will be replaced by plain HTML components (if possible).
            For example, plotly figures will be transformed into JPEG images,
            tabs widgets will be replaced by HTML '<details>' tag
        embed_into_suite : Union[Literal['suite'], Literal['suite-html-page'], None] , default None
            flag indicating that the 'CheckResult' output will be embedded
            into the 'SuiteResult' output. This flag tells the serializer
            how to serialize Plotly figures and whether navigation links
            should be included in the output.
            NOTE: this parameter does not have any effect if the 'full_html'
            parameter was set to 'True'

        Returns
        -------
        str
        """
        if full_html is True:
            embed_into_suite = None
        if embed_into_iframe is True:
            is_for_iframe_with_srcdoc = True

        sections_to_include = verify_include_parameter(check_sections)
        header, summary = self.prepare_header(output_id), self.prepare_summary()

        condition_table = (
            self.prepare_conditions_table(output_id=output_id)
            if 'condition-table' in sections_to_include
            else ''
        )
        additional_output = (
            self.prepare_additional_output(
                output_id=output_id,
                embed_into_iframe=embed_into_iframe,
                embed_into_suite=embed_into_suite,
                use_javascript=use_javascript,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )
            if 'additional-output' in sections_to_include
            else ''
        )

        content = f'{header}{summary}{condition_table}{additional_output}'
        content = f'<article data-name="check-result">{content}</article>'

        if embed_into_iframe is True:
            return self._serialize_to_iframe(content, use_javascript)

        if full_html is True:
            return self._serialize_to_full_html(content, use_javascript)

        if embed_into_suite is not None:
            return content

        if use_javascript is False:
            return f'<style>{DEEPCHECKS_STYLE}</style>{content}'

        if not contains_plots(self.value) or 'additional-output' not in sections_to_include:
            return f'{STYLE_LOADER}{content}'

        return f'{STYLE_LOADER}{PLOTLY_LOADER}{content}'

    def _serialize_to_iframe(self, content: str, use_javascript: bool = True) -> str:
        content = iframe_tag(
            title=self.value.get_header(),
            srcdoc=self._serialize_to_full_html(content, use_javascript),
            collapsible=False,
            clazz='deepchecks-resizable'
        )
        return (
            f'{STYLE_LOADER}{content}'
            if use_javascript
            else f'<style>{DEEPCHECKS_STYLE}</style>{content}'
        )

    def _serialize_to_full_html(self, content: str, use_javascript: bool = True) -> str:
        script = (
            f'<script type="text/javascript">{get_plotlyjs()}</script>'
            if use_javascript and contains_plots(self.value)
            else ''
        )
        return f"""
            <html>
            <head>
                <meta charset="utf-8"/>
                <style>{DEEPCHECKS_HTML_PAGE_STYLE}</style>
                {script}
            </head>
            <body>
            {content}
            </body>
            </html>
        """

    def prepare_header(self, output_id: t.Optional[str] = None) -> str:
        """Prepare the header section of the html output."""
        header = self.value.get_header()
        if output_id is not None:
            check_id = self.value.get_check_id(output_id)
            return f'<h3 id="{check_id}"><b>{header}</b></h3>'
        else:
            return f'<h3><b>{header}</b></h3>'

    def prepare_summary(self) -> str:
        """Prepare the summary section of the html output."""
        return f'<p>{self.value.get_metadata(with_doc_link=True)["summary"]}</p>'

    def prepare_conditions_table(
        self,
        max_info_len: int = 3000,
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> str:
        """Prepare the conditions table of the html output.

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
        str
        """
        if not self.value.have_conditions():
            return ''
        table = DataFrameHtmlSerializer(aggregate_conditions(
            self.value,
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id
        )).serialize()
        return f'<section data-name="conditions-table">{CommonHtml.conditions_summary_header}{table}</section>'

    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None,
        embed_into_iframe: bool = False,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        is_for_iframe_with_srcdoc: bool = False,
        use_javascript: bool = True,
    ) -> str:
        """Prepare the display content of the html output.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        embed_into_iframe : bool , default False
            whether to embed output into iframe or not
        embed_into_suite : Union[Literal['suite'], Literal['suite-html-page'], None] , default None
            flag indicating that the 'CheckResult' output will be embedded
            into the 'SuiteResult' output. This flag tells the serializer
            how to serialize Plotly figures and whether navigation links
            should be included in the output.
            This parameter does not have any effect if the 'full_html'
            parameter was set to 'True'
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not
        use_javascript : bool , default True
            whether to use  javascript in an output or not.
            If set to 'False', all components that require javascript
            to work will be replaced by plain HTML components (if possible).
            For example, plotly figures will be transformed into JPEG images,
            tabs widgets will be replaced by HTML '<details>' tag

        Returns
        -------
        str
        """
        output = [CommonHtml.additional_output_header]

        if len(self.value.display) == 0:
            output.append(CommonHtml.empty_content_placeholder)
        else:
            output.append(''.join(self._display_serializer.serialize(
                output_id=output_id,
                embed_into_iframe=embed_into_iframe,
                embed_into_suite=embed_into_suite,
                use_javascript=use_javascript,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )))

        if (
            output_id is not None
            and embed_into_suite is not None  # whether was embedded into suite output
        ):
            output.append(go_to_top_link(
                output_id=output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

        return f'<section data-name="additional-output">{"".join(output)}</section>'


class DisplayItemsSerializer(ABCDisplayItemsSerializer[str]):
    """CheckResult display items serializer."""

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        is_for_iframe_with_srcdoc: bool = False,
        embed_into_iframe: bool = False,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        use_javascript: bool = True,
        **kwargs
    ) -> t.List[str]:
        """Serialize CheckResult display items into HTML.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not
        embed_into_iframe : bool , default False
            whether to embed output into iframe or not
        embed_into_suite : Union[Literal['suite'], Literal['suite-html-page'], None] , default None
            flag indicating that the 'CheckResult' output will be embedded
            into the 'SuiteResult' output. This flag tells the serializer
            how to serialize Plotly figures and whether navigation links
            should be included in the output.
            This parameter does not have any effect if the 'full_html'
            parameter was set to 'True'
        use_javascript : bool , default True
            whether to use  javascript in an output or not.
            If set to 'False', all components that require javascript
            to work will be replaced by plain HTML components (if possible).
            For example, plotly figures will be transformed into JPEG images,
            tabs widgets will be replaced by HTML '<details>' tag

        Returns
        -------
        List[str]
        """
        return self.handle_display(
            self.value,
            output_id=output_id,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc,
            embed_into_iframe=embed_into_iframe,
            embed_into_suite=embed_into_suite,
            use_javascript=use_javascript,
            **kwargs
        )

    def handle_string(self, item, index, **kwargs) -> str:
        """Handle textual item."""
        return f'<div>{item}</div>'

    def handle_dataframe(self, item, index, **kwargs) -> str:
        """Handle dataframe item."""
        return DataFrameHtmlSerializer(item).serialize()

    def handle_callable(self, item, index, **kwargs) -> str:
        """Handle callable."""
        return ''.join(read_matplot_figures_as_html(item))

    def handle_figure(
        self,
        item: BaseFigure,
        index: int,
        embed_into_iframe: bool = False,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        plotly_to_image: bool = False,
        use_javascript: bool = True,
        **kwargs
    ) -> str:
        """Handle plotly figure item."""
        if use_javascript is False or plotly_to_image is True:
            return figure_to_html_image_tag(item)

        if embed_into_iframe or embed_into_suite == 'suite-html-page':
            return to_html(
                item,
                auto_play=False,
                include_plotlyjs=False,
                full_html=False,
                validate=True,
            )

        if embed_into_suite is None or embed_into_suite == 'suite':
            return figure_to_html(item)

        raise ValueError(f'Unknown "embed_into_suite" parameter value - {embed_into_suite}')

    def handle_display_map(
        self,
        item: 'check_types.DisplayMap',
        index: int,
        use_javascript: bool = True,
        **kwargs
    ) -> str:
        """Handle display map instance item."""
        if use_javascript is True:
            return tabs_widget({
                name: self.handle_display(
                    display_items,
                    use_javascript=use_javascript,
                    **kwargs
                )
                for name, display_items in item.items()
            })
        else:
            return ''.join(
                details_tag(
                    title=name,
                    attrs='class="deepchecks-collapsible"',
                    content_attrs='class="deepchecks-collapsible-content"',
                    content=''.join(self.handle_display(
                        display_items,
                        use_javascript=use_javascript,
                        **kwargs
                    ))
                )
                for name, display_items in item.items()
            )


def verify_include_parameter(
    include: t.Optional[t.Sequence[CheckResultSection]] = None
) -> t.Set[CheckResultSection]:
    """Verify CheckResultSection sequence."""
    sections = t.cast(
        t.Set[CheckResultSection],
        {'condition-table', 'additional-output'}
    )

    if include is None:
        sections_to_include = sections
    elif len(include) == 0:
        raise ValueError('include parameter cannot be empty')
    else:
        sections_to_include = set(include)

    if len(sections_to_include.difference(sections)) > 0:
        raise ValueError(
            'include parameter must contain '
            'Union[Literal["condition-table"], Literal["additional-output"]]'
        )

    return sections_to_include
