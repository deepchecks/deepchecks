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
import textwrap
import typing as t
import htmlmin

from plotly.basedatatypes import BaseFigure
from plotly.io import to_html
from plotly.offline.offline import get_plotlyjs
from typing_extensions import Literal as L

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import ABCDisplayItemsHandler, HtmlSerializer
from deepchecks.core.serialization.common import (aggregate_conditions, form_output_anchor,
                                                  plotly_loader_script, figure_creation_script,
                                                  DEEPCHECKS_STYLE,
                                                  DEEPCHECKS_HTML_PAGE_STYLE)
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer as DataFrameHtmlSerializer
from deepchecks.utils.html import imagetag, linktag
from deepchecks.utils.strings import get_random_string

__all__ = ['CheckResultSerializer']


CheckResultSection = t.Union[
    L['condition-table'],
    L['additional-output'],
]


EmbedmentWay = t.Union[
    L['suite'],            # embedded into suite display 
    L['suite-html-page'],  # embedded into suite display serialized with 'full_html' flag set to True
    None                   # was not embedded
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
        # connected: bool = True,
        embedment_way: EmbedmentWay = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
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
        connected : bool, default True
            whether to use CDN to load js libraries or to inject their code into output
        plotly_to_image : bool, default False
            whether to transform Plotly figure instance into static image or not
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        str
        """
        if full_html is True:
            embedment_way = None

        sections_to_include = verify_include_parameter(check_sections)
        sections = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(''.join(self.prepare_conditions_table(
                output_id=output_id
            )))

        if 'additional-output' in sections_to_include:
            sections.append(''.join(self.prepare_additional_output(
                output_id=output_id,
                embedment_way=embedment_way,
                plotly_to_image=plotly_to_image,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )))

        if full_html is False:
            return ''.join((
                f'<style>{DEEPCHECKS_STYLE}</style>',
                plotly_loader_script() if embedment_way is None else '', 
                *sections
            ))

        return htmlmin.minify(f"""
            <html>
            <head>
                <meta charset="utf-8"/>
                <style>{DEEPCHECKS_HTML_PAGE_STYLE}</style>
                <script type="text/javascript">{get_plotlyjs()}</script>
            </head>
            <body style="background-color: white;">
            {''.join(sections)}
            </body>
            </html>
        """)

    def prepare_header(self, output_id: t.Optional[str] = None) -> str:
        """Prepare the header section of the html output."""
        header = self.value.get_header()
        if output_id is not None:
            check_id = self.value.get_check_id(output_id)
            return f'<h4 id="{check_id}"><b>{header}</b></h4>'
        else:
            return f'<h4><b>{header}</b></h4>'

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
        return f'<h5><b>Conditions Summary</b></h5>{table}'

    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None,
        embedment_way: EmbedmentWay = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
    ) -> t.List[str]:
        """Prepare the display content of the html output.

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
        str
        """
        return DisplayItemsHandler.handle_display(
            self.value.display,
            output_id=output_id,
            embedment_way=embedment_way,
            plotly_to_image=plotly_to_image,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        )


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
        output.extend(super().handle_display(display, output_id=output_id, **kwargs))

        if len(display) == 0:
            output.append(cls.empty_content_placeholder())

        if output_id is not None and include_trailing_link:
            output.append(cls.go_to_top_link(
                output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

        return output

    @classmethod
    def header(cls) -> str:
        """Return header section."""
        return '<h5><b>Additional Outputs</b></h5>'

    @classmethod
    def empty_content_placeholder(cls) -> str:
        """Return placeholder in case of content absence."""
        return '<p><b>&#x2713;</b>Nothing to display</p>'

    @classmethod
    def go_to_top_link(
        cls,
        output_id: str,
        is_for_iframe_with_srcdoc: bool,
    ) -> str:
        """Return 'Go To Top' link."""
        link = linktag(
            text='Go to top',
            style={'font-size': '14px'},
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
            tags.append(imagetag(buffer.read()))
            buffer.close()

        return ''.join(tags)

    @classmethod
    def handle_figure(
        cls,
        item: BaseFigure,
        index: int,
        embedment_way: EmbedmentWay = None,
        plotly_to_image: bool = False,
        **kwargs
    ) -> str:
        """Handle plotly figure item."""
        if plotly_to_image is True:
            img = item.to_image(format='jpeg', engine='auto')
            return imagetag(img)
        
        if embedment_way == 'suite-html-page':
            return to_html(
                item,
                auto_play=False,
                include_plotlyjs=False,
                full_html=False,
                default_width='100%',
                default_height=525,
                validate=True,
            )
        elif embedment_way in {'suite', None}:
            container_id = f'deepchecks-{get_random_string(n=25)}'
            script = figure_creation_script(
                item,
                div_id=container_id,
                post_script=PLOTLY_POST_SCRIPT,
                default_width='100%',
                default_height=525,
                validate=True,
            )
            return (
                f'<div><div id="{container_id}"></div>'
                f'<script type="text/javascript">{script}</script></div>'
            )
        else:
            raise ValueError(f'Unknown "embedment_way" parameter value - {embedment_way}')

    @classmethod
    def handle_display_map(cls, item: 'check_types.DisplayMap', index: int, **kwargs) -> str:
        """Handle display map instance item."""
        template = textwrap.dedent("""
            <details class="deepchecks">
                <summary><strong>{name}</strong></summary>
                <div style="
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                    padding: 1.5rem;
                ">
                {content}
                </div>
            </details>
        """)
        return ''.join([
            template.format(
                name=k,
                content=''.join(cls.handle_display(
                    v,
                    include_header=False,
                    include_trailing_link=False,
                    **kwargs
                ))
            )
            for k, v in item.items()
        ])


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


PLOTLY_POST_SCRIPT = """
var gd = document.getElementById('{plot_id}');
var x = new MutationObserver(function (mutations, observer) {
    var display = window.getComputedStyle(gd).display;
    if (!display || display === 'none') {
        console.log([gd, 'removed!']);
        Plotly.purge(gd);
        observer.disconnect();
    }
});
/*Listen for the removal of the full notebook cells*/
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {
    x.observe(notebookContainer, {childList: true});
}
/* Listen for the clearing of the current output cell */
var outputEl = gd.closest('.output');
if (outputEl) {
    x.observe(outputEl, {childList: true});
}
"""