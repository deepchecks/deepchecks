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
import textwrap

import plotly.io as pio
from typing_extensions import Literal

from deepchecks.utils.strings import get_docs_summary
from deepchecks.utils.html import imagetag
from deepchecks.core.check_result import CheckResult
from deepchecks.core.check_result import TDisplayItem
from deepchecks.core.serialization.abc import HtmlSerializer
from deepchecks.core.serialization.abc import ABCDisplayItemsHandler
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer as DataFrameHtmlSerializer
from deepchecks.core.serialization.common import aggregate_conditions
from deepchecks.core.serialization.common import form_output_anchor
from deepchecks.core.serialization.common import form_check_id
from deepchecks.core.serialization.common import plotly_activation_script
from deepchecks.core.serialization.common import REQUIREJS_CDN


__all__ = ['CheckResultSerializer']


CheckResultSection = t.Union[
    Literal['condition-table'],
    Literal['additional-output'],
]


class CheckResultSerializer(HtmlSerializer[CheckResult]):
    """Serializes any CheckResult instance into HTML format.

    Parameters
    ----------
    value : CheckResult
        CheckResult instance that needed to be serialized.
    """

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[CheckResultSection]] = None,
        full_html: bool = False,
        include_plotlyjs: bool = True,
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
        include_plotlyjs : bool, default True
            whether to include plotlyjs activation script into output or not

        Returns
        -------
        str
        """
        sections_to_include = verify_include_parameter(check_sections)
        sections = [self.prepare_header(output_id), self.prepare_summary()]

        if 'condition-table' in sections_to_include:
            sections.append(''.join(self.prepare_conditions_table(output_id=output_id)))

        if 'additional-output' in sections_to_include:
            sections.append(''.join(self.prepare_additional_output(output_id)))

        if full_html is False and include_plotlyjs is False:
            return ''.join(sections)

        if full_html is False and include_plotlyjs is True:
            return ''.join([plotly_activation_script(), *sections])

        # TODO: use some style to make it prety
        return textwrap.dedent(f"""
            <html>
            <head><meta charset="utf-8"/></head>
            <body>
                {REQUIREJS_CDN}
                {plotly_activation_script()}
                {''.join(sections)}
            </body>
            </html>
        """)

    def prepare_header(self, output_id: t.Optional[str] = None) -> str:
        """Prepare the header section of the html output."""
        header = self.value.get_header()
        header = f'<b>{header}</b>'
        if output_id is not None:
            check_id = form_check_id(self.value.check, output_id)
            return f'<h4 id="{check_id}">{header}</h4>'
        else:
            return f'<h4>{header}</h4>'

    def prepare_summary(self) -> str:
        """Prepare the summary section of the html output."""
        summary = get_docs_summary(self.value.check)
        return f'<p>{summary}</p>'

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
        return f'<h5>Conditions Summary</h5>{table}'

    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None
    ) -> t.List[str]:
        """Prepare the display content of the html output.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        str
        """
        return DisplayItemsHandler.handle_display(
            self.value.display,
            output_id=output_id
        )


class DisplayItemsHandler(ABCDisplayItemsHandler):
    """Auxiliary class to decouple display handling logic from other functionality."""

    @classmethod
    def handle_display(
        cls,
        display: t.List[TDisplayItem],
        output_id: t.Optional[str] = None,
        **kwargs
    ) -> t.List[str]:
        """Serialize CheckResult display items into HTML.

        Parameters
        ----------
        display : List[Union[Callable, str, DataFrame, Styler]]
            list of display items
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links

        Returns
        -------
        List[str]
        """
        output = [
            cls.header(),
            *super().handle_display(display, **{'output_id': output_id, **kwargs})
        ]

        if len(display) == 0:
            output.append(cls.empty_content_placeholder())

        if output_id is not None:
            output.append(cls.go_to_top_link(output_id))

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
    def go_to_top_link(cls, output_id: str) -> str:
        """Return 'Go To Top' link."""
        href = form_output_anchor(output_id)
        return f'<br><a href="#{href}" style="font-size: 14px">Go to top</a>'

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
    def handle_figure(cls, item, index, **kwargs) -> str:
        """Handle plotly figure item."""
        bundle = pio.renderers['notebook'].to_mimebundle(item)
        return bundle['text/html']


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
