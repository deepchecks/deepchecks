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
import os
import typing as t
import warnings

from IPython.display import display
from ipywidgets import (HTML, Accordion, Box, Button, Checkbox, Dropdown, FloatText, IntProgress, Layout, Valid, VBox,
                        Widget)
import solara
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
from deepchecks.utils.ipython import create_progress_bar
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
            ),
            self.prepare_fixes()
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

    def prepare_fixes(self):
        # TODO: Change the if from getattr to isinstance, we left it like this because we had circular imports from
        # FixMixin import
        fixable_check_results = [check_result for check_result in self.value.get_not_passed_checks()
                                 if getattr(check_result.check, 'fix', None) is not None]
        accordion_name = "Fix Datasets!"
        if len(fixable_check_results) == 0:
            accordion = normalize_widget_style(Accordion(
                children=(HTML(value='<p>No fixes found.</p>'),),
                _titles={'0': accordion_name},
                selected_index=None
            ))
            return VBox(children=(
                # by putting HTML before the results accordion
                # we create a gap between them`s, failures section does not have
                # `section_anchor`` but we need to create a gap.
                # Take a look at the `prepare_results` method to understand
                HTML(value=''),
                accordion,
            ))

        # This is used so that widgets are set to the correct size automatically
        layout = Layout(width='auto', height='auto')

        # List of boxes of each check's check box and inputs
        check_and_inputs_boxes = []

        # This is important in order to distinct between train / test / both datasets when we fix duplicates
        # and also when we present the available fixes to the user
        train_result_to_checkbox = dict()
        test_result_to_checkbox = dict()
        train_test_result_to_checkbox = dict()

        input_widgets = dict(dict())

        # Iterate over all results, generate a checkbox for each result, and input widgets for its parameters for the
        # fix method
        for fixable_result in fixable_check_results:
            input_widget_boxes = []
            check, check_name = fixable_result.check, fixable_result.check.name()
            check_check_box = Checkbox(value=True, disabled=False, indent=False)

            # Determine if the check is related to a specific dataset and set its name accordingly
            if fixable_result.header is not None:
                if 'Train Dataset' in fixable_result.header:
                    check_name = check_name + ' - Train Dataset'
                    train_result_to_checkbox[fixable_result] = check_check_box
                elif 'Test Dataset' in fixable_result.header:
                    check_name = check_name + ' - Test Dataset'
                    test_result_to_checkbox[fixable_result] = check_check_box
                else:
                    train_test_result_to_checkbox[fixable_result] = check_check_box
            # If the header is None, we assume that the check is not related to a specific dataset
            else:
                train_test_result_to_checkbox[fixable_result] = check_check_box
            check_check_box.description = check_name

            input_widgets[check_name] = dict()
            for param_name, param_dict in check.fix_params.items():
                param_name_user_display = param_dict['display']
                param_input_widget = None
                params_description = None
                if type(param_dict['params']) == list:
                    # params_display is what the user sees, params is the value that is passed to the fix function
                    options = list(zip(param_dict['params_display'], param_dict['params']))
                    param_input_widget = Dropdown(
                        options=options,
                        value=options[0][1],
                        description=param_name_user_display,
                        style={'description_width': 'initial'}

                    )
                    params_description = zip(param_dict['params_display'], param_dict['params_description'])
                    params_description = [t[0] + " - " + t[1] for t in params_description]
                    params_description = "\n".join(params_description)

                elif param_dict['params'] == float:
                    param_input_widget = FloatText(
                        value=param_dict['params_display'],
                        disabled=False,
                        description=param_name_user_display,
                        style={'description_width': 'initial'}
                    )
                    params_description = param_dict['params_description']
                elif param_dict['params'] == bool:
                    param_input_widget = Checkbox(
                        value=param_dict['params_display'],
                        disabled=False,
                        description=param_name_user_display,
                        style={'description_width': 'initial'}
                    )
                    params_description = param_dict['params_description']
                input_widgets[check_name][param_name] = param_input_widget
                # Box the input widget to its description
                param_input_description = HTML(value="<b title='" + params_description + "'>ⓘ</b>",
                                               placeholder='widget description')
                param_input_box = Box([param_input_widget, param_input_description])

                # This is the product of this for loop, all the checks input widgets and their descriptions
                input_widget_boxes.append(param_input_box)

            # Place all input widgets in a vertical box
            input_widgets_vbox = VBox(children=input_widget_boxes, layout=layout)
            # Box the check's checkbox and its input widgets
            check_and_inputs_boxes.append(Box([check_check_box, input_widgets_vbox]))

        checks_vbox = VBox(children=check_and_inputs_boxes, layout=layout)

        def on_save_button_click(b):
            b.disabled = True
            b.description = 'Saving...'

            p_bar = IntProgress(
                value=0,
                min=0,
                max=2,
                step=1,
                description='Saving:',
                bar_style='success',
                orientation='horizontal'
            )
            display(p_bar)
            self.value.context.train.data.to_csv('train.csv')
            p_bar.value += 1
            self.value.context.test.data.to_csv('test.csv')
            p_bar.value += 1
            p_bar.close()

            b.description = 'Saved!'

            train_path = os.path.join(os.getcwd(), 'train.csv')
            test_path = os.path.join(os.getcwd(), 'test.csv')
            display(Valid(
                value=True,
                description=train_path,
                style={'description_width': 'initial'}

            ))
            display(Valid(
                value=True,
                description=test_path,
                style={'description_width': 'initial'}

            ))

        save_button = Button(
            description='Save datasets (CSV)',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Save as CSV',
            icon='download',  # (FontAwesome names without the `fa-` prefix)
            style={'description_width': 'initial'},
            layout=layout
        )
        save_button.on_click(on_save_button_click)

        input_validation_errors = []
        def input_validation(param_dict, value, current_check_name):
            if 'min_value' in param_dict:
                if value < param_dict['min_value']:
                    invalid_widget = Valid(
                        value=False,
                        description=current_check_name + ' - ' + param_dict['display'] + ' must be greater than ' +
                                    str(param_dict['min_value']),
                        style={'description_width': 'initial'},
                    )
                    input_validation_errors.append(invalid_widget)
                    display(invalid_widget)
                    return False

            if 'max_value' in param_dict:
                if value > param_dict['max_value']:
                    invalid_widget = Valid(
                        value=False,
                        description=current_check_name + " - " + param_dict['display'] + ' must be less than ' +
                                    str(param_dict['max_value']),
                        style={'description_width': 'initial'},
                    )
                    input_validation_errors.append(invalid_widget)
                    display(invalid_widget)
                    return False
            return True

        def on_fix_button_click(b):
            b.disabled = True
            b.description = 'Fixing...'
            check_name_to_params = dict()
            check_name_to_result = dict()

            for result, checkbox in test_result_to_checkbox.items():
                if checkbox.value:
                    check_name = result.check.name() + ' - Test Dataset'
                    check_name_to_params[check_name] = dict()

                    for param_name, param_widget in input_widgets[check_name].items():
                        if not input_validation(result.check.fix_params[param_name], param_widget.value, check_name):
                            b.disabled = False
                            b.description = 'Fix'
                            return
                        check_name_to_params[check_name][param_name] = param_widget.value
                    check_name_to_params[check_name].update({'dataset_kind': DatasetKind.TEST})
                    check_name_to_result[check_name] = result
            for result, checkbox in train_result_to_checkbox.items():
                if checkbox.value:
                    check_name = result.check.name() + ' - Train Dataset'
                    check_name_to_params[check_name] = dict()

                    for param_name, param_widget in input_widgets[check_name].items():
                        if not input_validation(result.check.fix_params[param_name], param_widget.value, check_name):
                            b.disabled = False
                            b.description = 'Fix'
                            return
                        check_name_to_params[check_name][param_name] = param_widget.value
                    check_name_to_params[check_name].update({'dataset_kind': DatasetKind.TEST})
                    check_name_to_result[check_name] = result
            for result, checkbox in train_test_result_to_checkbox.items():
                if checkbox.value:
                    check_name = result.check.name()
                    check_name_to_params[check_name] = dict()

                    for param_name, param_widget in input_widgets[check_name].items():
                        if not input_validation(result.check.fix_params[param_name], param_widget.value, check_name):
                            b.disabled = False
                            b.description = 'Fix'
                            return
                        check_name_to_params[check_name][param_name] = param_widget.value
                    check_name_to_result[check_name] = result
            for input_validation_widget in input_validation_errors:
                input_validation_widget.close()
            p_bar = IntProgress(
                value=0,
                min=0,
                max=len(check_name_to_result),
                step=1,
                description='Fixing :',
                bar_style='success',
                orientation='horizontal',
                style={'description_width': 'initial'}
            )
            display(p_bar)
            current_index = 1
            for check_name, input_params in check_name_to_params.items():
                for result in self.value.results:
                    if result.check.name() in check_name:
                        p_bar.description = 'Fixing ' + check_name + ' ' + str(current_index) + '/' + str(
                            len(check_name_to_params))
                        check_name_to_result[check_name].check.fix_logic(context=self.value.context,
                                                                         check_result=check_name_to_result[check_name],
                                                                         **input_params)
                        p_bar.value += 1
                        current_index += 1
                        break
            p_bar.close()

            display(Valid(
                value=True,
                description='Done Fixing!',
            ))
            #display(save_button)
            display(solara.FileDownload(data=self.value.context.train.data.to_csv(), filename="fixed_train.csv"))
            display(solara.FileDownload(data=self.value.context.train.data.to_csv(), filename="fixed_train.csv"))

            b.description = 'Fixed!'

        fix_button = Button(
            description='Fix!',
            disabled=False,
            button_style='info',
            tooltip='Fix!',
            icon='wrench'
        )
        fix_button.on_click(on_fix_button_click)

        box = Box(children=[checks_vbox])
        vbox = VBox(children=[box, fix_button])
        accordion = normalize_widget_style(Accordion(
            children=(VBox(children=[vbox]),),
            _titles={'0': accordion_name},
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


def select_serializer(result):
    if isinstance(result, check_types.CheckResult):
        return CheckResultWidgetSerializer(result)
    elif isinstance(result, check_types.CheckFailure):
        return CheckFailureWidgetSerializer(result)
    else:
        raise TypeError(f'Unknown type of result - {type(result)}')
