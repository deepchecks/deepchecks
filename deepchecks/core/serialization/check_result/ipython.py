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
"""Module containing CheckResult serialization logic."""
import typing as t

from IPython.display import HTML, Image
from plotly.basedatatypes import BaseFigure

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import IPythonFormatter, IPythonSerializer
from deepchecks.core.serialization.common import flatten
from deepchecks.core.serialization.dataframe.html import DataFrameSerializer

from . import html

__all__ = ['CheckResultSerializer']


class CheckResultSerializer(IPythonSerializer['check_types.CheckResult']):
    """Serializes any CheckResult instance into a list of IPython formatters.

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
        self._html_serializer = html.CheckResultSerializer(value)

    def serialize(
        self,
        output_id: t.Optional[str] = None,
        check_sections: t.Optional[t.Sequence[html.CheckResultSection]] = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False,
        **kwargs
    ) -> t.List[IPythonFormatter]:
        """Serialize a CheckResult instance into a list of IPython formatters.

        Parameters
        ----------
        output_id : Optional[str], default None
            unique output identifier that will be used to form anchor links
        check_sections : Optional[Sequence[Literal['condition-table', 'additional-output']]], default None
            sequence of check result sections to include into the output,
            in case of 'None' all sections will be included
        plotly_to_image : bool, default False
            whether to transform Plotly figure instance into static image or not
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        List[IPythonFormatter]
        """
        sections_to_include = html.verify_include_parameter(check_sections)
        sections: t.List[IPythonFormatter] = [
            self.prepare_header(output_id),
            self.prepare_summary()
        ]

        if 'condition-table' in sections_to_include:
            sections.append(self.prepare_conditions_table(
                output_id=output_id
            ))

        if 'additional-output' in sections_to_include:
            sections.extend(self.prepare_additional_output(
                output_id=output_id,
                plotly_to_image=plotly_to_image,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            ))

        return list(flatten(sections))

    def prepare_header(self, output_id: t.Optional[str] = None) -> HTML:
        """Prepare the header section."""
        return HTML(self._html_serializer.prepare_header(output_id=output_id))

    def prepare_summary(self) -> HTML:
        """Prepare the summary section."""
        return HTML(self._html_serializer.prepare_summary())

    def prepare_conditions_table(
        self,
        max_info_len: int = 3000,
        include_icon: bool = True,
        include_check_name: bool = False,
        output_id: t.Optional[str] = None,
    ) -> HTML:
        """Prepare the conditions table.

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
        HTML
        """
        return HTML(self._html_serializer.prepare_conditions_table(
            max_info_len=max_info_len,
            include_icon=include_icon,
            include_check_name=include_check_name,
            output_id=output_id,
        ))

    def prepare_additional_output(
        self,
        output_id: t.Optional[str] = None,
        plotly_to_image: bool = False,
        is_for_iframe_with_srcdoc: bool = False
    ) -> t.List[IPythonFormatter]:
        """Prepare the display content.

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
        List[IPythonFormatter]
        """
        return DisplayItemsHandler.handle_display(
            self.value.display,
            output_id=output_id,
            plotly_to_image=plotly_to_image,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
        )


class DisplayItemsHandler(html.DisplayItemsHandler):
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
    ) -> t.List[IPythonFormatter]:
        """Serialize CheckResult display items into IPython displayable objects.

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
        List[IPythonFormatter]
        """
        return list(flatten(super().handle_display(
            display=display,
            output_id=output_id,
            include_header=include_header,
            include_trailing_link=include_trailing_link,
            is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc,
            **kwargs
        )))

    @classmethod
    def header(cls):
        """Return header section."""
        return HTML(super().header())

    @classmethod
    def empty_content_placeholder(cls):
        """Return placeholder in case of content absence."""
        return HTML(super().empty_content_placeholder())

    @classmethod
    def go_to_top_link(cls, output_id: str, is_for_iframe_with_srcdoc: bool):
        """Return 'Go To Top' link."""
        return HTML(super().go_to_top_link(output_id, is_for_iframe_with_srcdoc))

    @classmethod
    def handle_string(cls, item, index, **kwargs):
        """Handle textual item."""
        return HTML(super().handle_string(item, index, **kwargs))

    @classmethod
    def handle_dataframe(cls, item, index, **kwargs):
        """Handle dataframe item."""
        return HTML(DataFrameSerializer(item).serialize())

    @classmethod
    def handle_callable(cls, item, index, **kwargs):
        """Handle callable."""
        # NOTE:
        # we are calling `handle_callable` method not from 'html.DisplayItemsHandler'
        # but from 'abc.ABCDisplayItemsHandler' that returns list of byte streams
        #
        images = []
        figures = super(html.DisplayItemsHandler, cls).handle_callable(  # pylint: disable=bad-super-call
            item, index, **kwargs
        )
        for it in figures:
            it.seek(0)
            images.append(Image(data=it.read(), format='png'))
        return images

    @classmethod
    def handle_figure(
        cls,
        item: BaseFigure,
        index: int,
        plotly_to_image: bool = False,
        **kwargs
    ):
        """Handle plotly figure item."""
        return (
            item
            if not plotly_to_image
            else Image(data=item.to_image(format='jpeg', engine='auto'), format='jpeg')
        )

    @classmethod
    def handle_display_map(cls, item: 'check_types.DisplayMap', index: int, **kwargs):
        """Handle display map instance item."""
        level = kwargs.pop('_level', 0)
        content = []
        for name, display_items in item.items():
            content.append(HTML(f'<h5><b>{">"*level}{name}</b></h5>'))
            content.extend(cls.handle_display(
                display_items,
                include_header=False,
                include_trailing_link=False,
                _level=level+1,
                **kwargs
            ))
        return content
