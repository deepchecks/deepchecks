# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
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

from ipywidgets import HTML, Accordion, VBox, Widget, Checkbox, Dropdown, Box, Button, FloatText, Layout

from deepchecks.core import DatasetKind
from deepchecks.core import check_result as check_types
from deepchecks.core import suite
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
        # TODO: OFIR TESTING CODE DELETE THIS

        button = Button(
            description='Fix!',
            disabled=False,
            button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Fix!',
            icon='wrench'  # (FontAwesome names without the `fa-` prefix)
        )

        # TODO: Change the if from getattr to isinstance, we left it like this because we had circular imports from
        # FixMixin import
        fixable_check_results = [check_result for check_result in self.value.get_not_passed_checks()
                                 if getattr(check_result.check, 'fix', None) is not None]
        print("Found checks are: ", list([check_result.check.name() for check_result in fixable_check_results]))

        if len(fixable_check_results) == 0:
            print('No fixable check found')
            return

        checks_vbox_children = []
        input_widgets = dict(dict())
        check_to_checkbox = dict()
        layout = Layout(width='auto', height='40px')
        for fixable_result in fixable_check_results:
            check, check_name = fixable_result.check, fixable_result.check.name()
            print("Handling check: ", check_name)
            if check_name not in input_widgets:
                input_widgets[check_name] = dict()
            for param_name, param_dict in check.fix_params.items():
                user_display = param_dict['display']
                if type(param_dict['params']) == list:
                    options = list(zip(param_dict['params_display'], param_dict['params']))
                    dropdown = Dropdown(
                        options=options,
                        value=options[0][1],
                        description=user_display,
                    )
                    input_widgets[check_name][param_name] = dropdown
                elif param_dict['params'] == float:
                    input_widgets[check_name][param_name] = FloatText(
                        value=param_dict['params_display'],
                        description=user_display,
                        disabled=False
                    )
            vbox = VBox(children=list(input_widgets[check_name].values()))
            check_box = Checkbox(value=True, description=check_name, disabled=False, indent=False)
            check_to_checkbox[check] = check_box
            checks_vbox_children.append(Box([check_box, vbox]))
        checks_vbox = VBox(children=checks_vbox_children)
        save_button = Button(
            description='Save datasets',
            disabled=True,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Fix!',
            icon='download'  # (FontAwesome names without the `fa-` prefix)
        )

        def on_fix_button_click(b):
            print('Fixing!')
            for check, checkbox in check_to_checkbox.items():
                if checkbox.value:
                    print(f'Fixing {check.name()}')
            # data_duplicates_check.fix_logic(context=self.value.context, check_result=data_duplicates_check_result,
            # dataset_kind=DatasetKind.TRAIN, keep=dropdowns[data_duplicates_check.name()]['keep'].value)
            save_button.disabled = False

        def on_save_button_click(b):
            print('Saving!')
            self.value.context.train.data.to_csv('train.csv')
            self.value.context.test.data.to_csv('test.csv')
            print('saved!')

        button.on_click(on_fix_button_click)
        save_button.on_click(on_save_button_click)
        box = Box(children=[checks_vbox])
        another_vbox = VBox(children=[box, button, save_button])
        accordion = Accordion(children=[another_vbox], _titles={'0': 'Fixes'}, selected_index='0')

        content = VBox(children=[
            self.prepare_summary(output_id=output_id, **kwargs),
            *accordions, accordion
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

            serialized_results = [
                select_serializer(it).serialize(output_id=section_id, **kwargs)
                for it in results
                if it.display  # we do not form full-output for the check results without display
            ]

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
