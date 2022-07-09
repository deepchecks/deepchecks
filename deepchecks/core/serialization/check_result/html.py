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
from deepchecks.core.serialization.abc import ABCDisplayItemsHandler, HtmlSerializer
from deepchecks.core.serialization.common import (aggregate_conditions, figure_to_html, form_output_anchor,
                                                  plotly_loader_script)
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer as DataFrameHtmlSerializer
from deepchecks.utils.html import details_tag, iframe_tag, imagetag, linktag, tabs_widget

__all__ = ['CheckResultSerializer']


CheckResultSection = t.Union[
    L['condition-table'],
    L['additional-output'],
]


CheckEmbedmentWay = t.Union[
    L['suite'],            # embedded into suite display
    L['suite-html-page'],  # embedded into suite display serialized with 'full_html' flag set to True
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

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        full_html: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
        use_javascript: bool = True,
        embed_into_iframe: bool = False,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        **kwargs
    ) -> str:
        """Serialize a CheckResult instance into HTML format.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included
        full_html : bool, default False
            whether to return a fully independent HTML document or only CheckResult content
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

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

        content = f'<article class="deepchecks">{header}{summary}{condition_table}{additional_output}</article>'

        if full_html is False:
            if embed_into_iframe is True:
                return self._serialize_to_iframe(content, use_javascript)
            elif embed_into_suite is not None or use_javascript is False:
                return content
            else:
                return f'{plotly_loader_script()}{content}'

        return self._serialize_to_full_html(content)

    def _serialize_to_iframe(self, content: str, use_javascript: bool = True) -> str:
        content = iframe_tag(
            title=self.value.get_header(),
            srcdoc=self._serialize_to_full_html(content, use_javascript),
            collapsible=False
        )
        return f'<style>{DEEPCHECKS_STYLE}</style>{content}'

    def _serialize_to_full_html(self, content: str, use_javascript: bool = True) -> str:
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
            <body class="deepchecks">
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
        return f'<p>{self.value.get_metadata()["summary"]}</p>'

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
        return f'<section><h4><b>Conditions Summary</b></h4>{table}</section>'

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
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        str
        """
        content = ''.join(DisplayItemsHandler.handle_display(
            self.value.display,
            output_id=output_id,
            embed_into_iframe=embed_into_iframe,
            embed_into_suite=embed_into_suite,
            use_javascript=use_javascript,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        ))
        return f'<section>{content}</section>'


class DisplayItemsHandler(ABCDisplayItemsHandler):
    """Auxiliary class to decouple display handling logic from other functionality."""

    @classmethod
    def handle_display(
        cls,
        display: t.List['check_types.TDisplayItem'],
        output_id: t.Optional[str] = None,
        is_for_iframe_with_srcdoc: bool = False,
        include_header: bool = True,
        include_trailing_link: bool = True,
        embed_into_iframe: bool = False,
        embed_into_suite: t.Optional[CheckEmbedmentWay] = None,
        **kwargs
    ) -> t.List[str]:
        """Serialize CheckResult display items into HTML.

        Parameters
        ----------
        display : List[Union[str, DataFrame, Styler, BaseFigure, Callable, DisplayMap]]
            list of display items
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not
        include_header: bool, default True
            whether to include header
        include_trailing_link: bool, default True
            whether to include "go to top" link

        Returns
        -------
        List[str]
        """
        output = [cls.header()] if include_header else []

        output.extend(super().handle_display(
            display,
            output_id=output_id,
            embed_into_iframe=embed_into_iframe,
            embed_into_suite=embed_into_suite,
            **kwargs
        ))

        if len(display) == 0:
            output.append(cls.empty_content_placeholder())

        if (
            output_id is not None
            and include_trailing_link
            and embed_into_suite is not None  # whether was embedded into suite output
        ):
            output.append(cls.go_to_top_link(
                output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

        return output

    @classmethod
    def header(cls) -> str:
        """Return header section."""
        return '<h4><b>Additional Outputs</b></h4>'

    @classmethod
    def empty_content_placeholder(cls) -> str:
        """Return placeholder in case of content absence."""
        return '<p><b>&#x2713;</b>Nothing to display</p>'

    @classmethod
    def go_to_top_link(
        cls,
        output_id: str,
        is_for_iframe_with_srcdoc: bool
    ) -> str:
        """Return 'Go To Top' link."""
        link = linktag(
            text='Go to top',
            href=f'#{form_output_anchor(output_id)}',
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        )
        return f'<br>{link}'

    @classmethod
    def handle_string(cls, item, index, **kwargs) -> str:
        """Handle textual item."""
        return f'<div>{item}</div>'

    @classmethod
    def handle_dataframe(cls, item, index, **kwargs) -> str:
        """Handle dataframe item."""
        return DataFrameHtmlSerializer(item).serialize()

    @classmethod
    def handle_callable(cls, item, index, **kwargs) -> str:
        """Handle callable."""
        images = super().handle_callable(item, index, **kwargs)
        tags = []

        for buffer in images:
            buffer.seek(0)
            tags.append(imagetag(
                buffer.read(),
                prevent_resize=False,
                style='min-width: 300px; width: 70%; height: 100%;'
            ))
            buffer.close()

        return ''.join(tags)

    @classmethod
    def handle_figure(
        cls,
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
            img = item.to_image(format='jpeg', engine='auto')
            return imagetag(
                img,
                prevent_resize=False,
                style='min-width: 300px; width: 70%; height: 100%;'
            )

        if embed_into_iframe or embed_into_suite == 'suite-html-page':
            return to_html(
                item,
                auto_play=False,
                include_plotlyjs=False,
                full_html=False,
                default_width='100%',
                default_height=525,
                validate=True,
            )

        if embed_into_suite is None or embed_into_suite == 'suite':
            return figure_to_html(item)

        raise ValueError(f'Unknown "embed_into_suite" parameter value - {embed_into_suite}')

    @classmethod
    def handle_display_map(
        cls,
        item: 'check_types.DisplayMap',
        index: int,
        use_javascript: bool = True,
        **kwargs
    ) -> str:
        """Handle display map instance item."""
        if use_javascript is True:
            return tabs_widget({
                name: cls.handle_display(
                    display_items,
                    include_header=False,
                    include_trailing_link=False,
                    use_javascript=use_javascript,
                    **kwargs
                )
                for name, display_items in item.items()
            })
        else:
            return ''.join(
                details_tag(
                    title=name,
                    attrs='class="deepchecks"',
                    content=''.join(cls.handle_display(
                        display_items,
                        include_header=False,
                        include_trailing_link=False,
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
