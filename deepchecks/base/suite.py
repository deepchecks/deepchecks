"""Module containing the Suite object, used for running a set of checks together."""
# pylint: disable=protected-access,broad-except
from collections import OrderedDict
from typing import Union, List, Tuple

import pandas as pd
from IPython.core.display import display_html, display
from ipywidgets import IntProgress, HTML, VBox

from deepchecks.base.check import BaseCheck, CheckResult, TrainTestBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck

__all__ = ['CheckSuite', 'SuiteResult']

from deepchecks.utils import DeepchecksValueError, is_widgets_enabled


def get_display_exists_icon(exists: bool):
    if exists:
        return '<div style="text-align: center">Yes</div>'
    return '<div style="text-align: center">No</div>'


class SuiteResult:
    """Contain the results of a suite run."""

    name: str
    results: List[Union[CheckResult, Tuple]]

    def __init__(self, name: str, results):
        """Initialize suite result."""
        self.name = name
        self.results = results

    def _ipython_display_(self, only_summary=False):
        display_html(f'<h1>{self.name}</h1>', raw=True)
        conditions_table = []
        checks_without_condition_table = []
        errors_table = []

        for result in self.results:
            if isinstance(result, CheckResult):
                if result.have_conditions():
                    for cond_result in result.conditions_results:
                        sort_value = cond_result.get_sort_value()
                        icon = cond_result.get_icon()
                        conditions_table.append([icon, result.header, cond_result.name,
                                                 cond_result.details, sort_value])
                else:
                    checks_without_condition_table.append([result.header,
                                                           get_display_exists_icon(result.have_display())])
            elif isinstance(result, Tuple):
                errors_table.append(result)

        # First print summary
        display_html('<h2>Checks Summary</h2>', raw=True)
        if conditions_table:
            display_html('<h3>With Conditions</h3>', raw=True)
            table = pd.DataFrame(data=conditions_table, columns=['Status', 'Check', 'Condition', 'More Info', 'sort'])
            table.sort_values(by=['sort'], inplace=True)
            table.drop('sort', axis=1, inplace=True)
            SuiteResult._display_table(table)
        if checks_without_condition_table:
            display_html('<h3>Without Conditions</h3>', raw=True)
            table = pd.DataFrame(data=checks_without_condition_table, columns=['Check', 'Has Display?'])
            SuiteResult._display_table(table)
        if errors_table:
            display_html('<h3>With Error</h3>', raw=True)
            table = pd.DataFrame(data=errors_table, columns=['Check', 'Error'])
            SuiteResult._display_table(table)
        # If verbose print all displays
        if not only_summary:
            only_check_with_display = [r for r in self.results
                                       if isinstance(r, CheckResult) and r.have_display()]
            # If there are no checks with display doesn't print anything else
            if only_check_with_display:
                checks_not_passed = [r for r in only_check_with_display
                                     if r.have_conditions() and not r.passed_conditions()]
                checks_without_condition = [r for r in only_check_with_display
                                            if not r.have_conditions() and r.have_display()]
                checks_passed = [r for r in only_check_with_display
                                 if r.have_conditions() and r.passed_conditions() and r.have_display()]

                display_html('<hr><h2>Results Display</h2>', raw=True)
                if checks_not_passed:
                    display_html('<h3>Checks with Failed Condition</h3>', raw=True)
                    for result in sorted(checks_not_passed, key=lambda x: x.get_conditions_sort_value()):
                        result._ipython_display_()
                if checks_without_condition:
                    display_html('<h3>Checks without Condition</h3>', raw=True)
                    for result in checks_without_condition:
                        result._ipython_display_()
                if checks_passed:
                    display_html('<h3>Checks with Passed Condition</h3>', raw=True)
                    for result in checks_passed:
                        result._ipython_display_()

    @classmethod
    def _display_table(cls, df):
        df_styler = df.style
        df_styler.set_table_styles([dict(selector='th,td', props=[('text-align', 'left')])])
        df_styler.hide_index()
        display_html(df_styler.render(), raw=True)


class CheckSuite(BaseCheck):
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
    """

    checks: OrderedDict
    name: str
    _check_index: int

    def __init__(self, name: str, *checks):
        """Get 'Check's and 'CheckSuite's to run in given order."""
        super().__init__()
        self.name = name
        self.checks = OrderedDict()
        self._check_index = 0
        for check in checks:
            self.add(check)

    def run(self, model=None, train_dataset=None, test_dataset=None, check_datasets_policy: str = 'test') \
            -> SuiteResult:
        """Run all checks.

        Args:
          model: A scikit-learn-compatible fitted estimator instance
          train_dataset: Dataset object, representing data an estimator was fitted on
          test_dataset: Dataset object, representing data an estimator predicts on
          check_datasets_policy: str, one of either ['both', 'train', 'test'].
                                 Determines the policy by which single dataset checks are run when two datasets are
                                 given, one for train and the other for test.

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if check_datasets_policy not in ['both', 'train', 'test']:
            raise ValueError('check_datasets_policy must be one of ["both", "train", "test"]')

        # Create progress bar
        progress_bar = IntProgress(value=0, min=0, max=len(self.checks),
                                   bar_style='info', style={'bar_color': '#9d60fb'}, orientation='horizontal')
        label = HTML()
        box = VBox(children=[label, progress_bar])
        self._display_widget(box)

        # Run all checks
        results = []
        for name, check in self.checks.items():
            try:
                label.value = f'Running {str(check)}'
                if isinstance(check, TrainTestBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run(train_dataset=train_dataset, test_dataset=test_dataset,
                                                 model=model)
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, CompareDatasetsBaseCheck):
                    if train_dataset is not None and test_dataset is not None:
                        check_result = check.run(dataset=test_dataset, baseline_dataset=train_dataset,
                                                 model=model)
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, SingleDatasetBaseCheck):
                    if check_datasets_policy in ['both', 'train'] and train_dataset is not None:
                        check_result = check.run(dataset=train_dataset, model=model)
                        check_result.header = f'{check_result.header} - Train Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                    if check_datasets_policy in ['both', 'test'] and test_dataset is not None:
                        check_result = check.run(dataset=test_dataset, model=model)
                        check_result.header = f'{check_result.header} - Test Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, ModelOnlyBaseCheck):
                    if model is not None:
                        check_result = check.run(model=model)
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append((name, exp))
            progress_bar.value = progress_bar.value + 1

        progress_bar.close()
        label.close()
        box.close()

        return SuiteResult(self.name, results)

    def __repr__(self, tabs=0):
        """Representation of suite as string."""
        tabs_str = '\t' * tabs
        checks_str = ''.join([f'\n{c.__repr__(tabs + 1, str(n) + ": ")}' for n, c in self.checks.items()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, index):
        """Access check inside the suite by name."""
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        return self.checks[index]

    def add(self, check):
        """Add a check or a suite to current suite.

        Args:
            check (BaseCheck): A check or suite to add.
        """
        if not isinstance(check, BaseCheck):
            raise Exception(f'CheckSuite receives only `BaseCheck` objects but got: {check.__class__.__name__}')
        if isinstance(check, CheckSuite):
            for c in check.checks.values():
                self.add(c)
        else:
            self.checks[self._check_index] = check
            self._check_index += 1
        return self

    def remove(self, index: int):
        """Remove a check by given index.

        Args:
            index (int): Index of check to remove.
        """
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        self.checks.pop(index)
        return self

    def _display_widget(self, param):
        if is_widgets_enabled():
            display(param)
