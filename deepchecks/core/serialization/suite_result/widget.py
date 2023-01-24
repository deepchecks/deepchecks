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
"""Module containing ipywidget serializer for the SuiteResult type."""
import typing as t
import warnings
from collections import defaultdict

from ipywidgets import HTML, Accordion, Tab, VBox, Widget

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.checks import DatasetKind
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.check_failure.widget import CheckFailureSerializer as CheckFailureWidgetSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.core.serialization.common import Html as CommonHtml
from deepchecks.core.serialization.common import (aggregate_conditions, create_failures_dataframe,
                                                  create_results_dataframe, form_output_anchor, join,
                                                  normalize_widget_style)
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
        super().__init__(value=value)
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
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

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
                title='Didn\'t Pass',
                results=not_passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                **kwargs
            ),
            self.prepare_results(
                title='Passed',
                results=passed_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_conditions_summary,
                **kwargs
            ),
            self.prepare_results(
                title='Other',
                results=other_checks,
                output_id=output_id,
                summary_creation_method=self.prepare_unconditioned_results_summary,
                check_sections=['additional-output'],
                **kwargs
            ),
            self.prepare_failures(
                title='Didn\'t Run',
                failures=not_ran_checks,
                output_id=output_id,
                **kwargs
            )
        ]

        content = VBox(children=[
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
    ) -> VBox:
        """Prepare failures section.

        Parameters
        ----------
        failures : Sequence[CheckFailure]
            sequence of check failures
        title : str
            accordion title

        Returns
        -------
        ipywidgets.VBox
        """
        if len(failures) == 0:
            children = (HTML(value='<p>No outputs to show.</p>'),)
        else:
            styler = create_failures_dataframe(failures).style
            # style.hide_index() was deprecated in the latest versions and new method was added
            styler = styler.hide(axis='index') if hasattr(styler, 'hide') else styler.hide_index()
            table = DataFrameSerializer(styler).serialize()
            children = (table,)
        accordion = normalize_widget_style(Accordion(
            children=children,
            _titles={'0': title},
            selected_index=None
        ))
        return VBox(children=(
            # by putting `section_anchor` before the results accordion
            # we create a gap between them`s, failures section does not have
            # `section_anchor`` but we need to create a gap.
            # Take a look at the `prepare_results` method to understand
            HTML(value=''),
            accordion,
        ))

    def prepare_results(
        self,
        results: t.Sequence['check_types.CheckResult'],
        title: str,
        output_id: t.Optional[str] = None,
        summary_creation_method: t.Optional[t.Callable[..., Widget]] = None,
        **kwargs
    ) -> VBox:
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

        Returns
        -------
        ipywidgets.VBox
        """
        if len(results) == 0:
            section_anchor = HTML(value='')
            accordion = normalize_widget_style(Accordion(
                children=(HTML(value='<p>No outputs to show.</p>'),),
                _titles={'0': title},
                selected_index=None
            ))
        else:
            section_id = f'{output_id}-section-{get_random_string()}'
            section_anchor = HTML(value=f'<span id="{form_output_anchor(section_id)}"></span>')

            grouped_results: t.Dict[
                int,
                t.Dict[t.Union[None, DatasetKind], 'check_types.CheckResult']
            ] = defaultdict(dict)

            for it in results:
                if not it.display:
                    # we do not form full-output for the check results without display
                    continue

                if it._suite_execution_info is None:  # pylint: disable=protected-access
                    # internal error
                    raise ValueError(
                        "'CheckResult' instance that was not produced by a 'Suite' "
                        "instance was added to a 'SuiteResult' collection instance"
                    )

                check_index = it._suite_execution_info.check_unique_index  # pylint: disable=protected-access
                check_input_kind = it._suite_execution_info.check_input_kind  # pylint: disable=protected-access

                if (
                    check_index in grouped_results
                    and check_input_kind in grouped_results[check_index]
                ):
                    # internal error
                    raise ValueError('CheckResult duplication')

                # serialization_result = select_serializer(it).serialize(output_id=section_id, **kwargs)
                grouped_results[check_index][check_input_kind] = it

            serialized_results = []

            for check_index, check_results in grouped_results.items():
                if len(check_results) == 1:
                    check_result = list(check_results.values())[0]
                    serialized_result = select_serializer(check_result).serialize(output_id=section_id, **kwargs)
                    serialized_results.append(serialized_result)

                elif len(check_results) == 2:
                    tab = Tab()
                    children = []
                    anchor_ids = []

                    for (i, (k, v)) in enumerate(check_results.items()):
                        tab.set_title(i, t.cast(DatasetKind, k).value)
                        anchor_ids.append(v.get_check_id(section_id))
                        children.append(select_serializer(v).serialize(**kwargs))

                    anchor_tags = ''.join(
                        f"<h6 id='{id}'></h6>"
                        for id in anchor_ids
                    )

                    tab.children = children
                    serialized_results.append(VBox(children=[HTML(value=anchor_tags), tab]))

                else:
                    # internal error
                    raise ValueError('Unexpected number of widgets')

            if callable(summary_creation_method):
                children = (
                    summary_creation_method(results=results, output_id=section_id, **kwargs),
                    HTML(value=CommonHtml.light_hr),
                    *join(serialized_results, HTML(value=CommonHtml.light_hr))
                )
            else:
                children = (
                    *join(serialized_results, HTML(value=CommonHtml.light_hr)),
                )

            accordion = normalize_widget_style(Accordion(
                children=(VBox(children=children),),
                _titles={'0': title},
                selected_index=None
            ))

        return VBox(children=(
            # "go to top" link should bring the user a bit higher,
            # to the top of the accordion, enabling easier folding,
            # therefore we need to put section_anchor before the accordion
            section_anchor,
            accordion
        ))

    def prepare_conditions_summary(
        self,
        results: t.Sequence['check_types.CheckResult'],
        output_id: t.Optional[str] = None,
        include_check_name: bool = True,
        is_for_iframe_with_srcdoc: bool = False,
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
        is_for_iframe_with_srcdoc : bool, default False
            anchor links, in order to work within iframe require additional prefix
            'about:srcdoc'. This flag tells function whether to add that prefix to
            the anchor links or not

        Returns
        -------
        ipywidgets.Widget
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
    ) -> Widget:
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
        ipywidgets.Widget
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df = create_results_dataframe(
                results=results,
                output_id=output_id,
                is_for_iframe_with_srcdoc=is_for_iframe_with_srcdoc
            )
            return DataFrameSerializer(df.style.hide_index()).serialize()


def select_serializer(result):
    if isinstance(result, check_types.CheckResult):
        return CheckResultWidgetSerializer(result)
    elif isinstance(result, check_types.CheckFailure):
        return CheckFailureWidgetSerializer(result)
    else:
        raise TypeError(f'Unknown type of result - {type(result)}')
