"""Module containing the Suite object, used for running a set of checks together."""
# pylint: disable=protected-access,broad-except
from collections import OrderedDict
from typing import Union, List, Tuple

import pandas as pd
from IPython.core.display import display_html, display
from ipywidgets import IntProgress, HTML, VBox

from mlchecks.base.check import BaseCheck, CheckResult, TrainValidationBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck

__all__ = ['CheckSuite', 'SuiteResult']

from mlchecks.utils import is_notebook


class SuiteResult:
    """Contain the results of a suite run."""

    name: str
    results: List[Union[CheckResult, 'SuiteResult']]

    def __init__(self, name: str, results):
        """Initialize suite result."""
        self.name = name
        self.results = results

    def _ipython_display_(self, only_summary=False):
        display_html(f'<h2>{self.name}</h2>', raw=True)
        conditions_table = []
        errors_table = []
        for result in self.results:
            if isinstance(result, CheckResult):
                for cond_result in result.conditions_results:
                    sort_value, icon = cond_result.get_status()
                    conditions_table.append([icon, result.header, cond_result.name,
                                             cond_result.details, sort_value])
            elif isinstance(result, Tuple):
                errors_table.append(result)

        # First print summary
        if conditions_table:
            display_html('<h3>Conditions Summary</h3>', raw=True)
            table = pd.DataFrame(data=conditions_table, columns=['Pass?', 'Check', 'Condition', 'More Info', 'sort'])
            table.sort_values(by=['sort'], inplace=True)
            table.drop('sort', axis=1, inplace=True)
            SuiteResult._display_table(table)
        # If verbose print all displays
        if not only_summary:
            only_check_results = [r for r in self.results if isinstance(r, CheckResult)]
            checks_not_passed = [r for r in only_check_results if r.have_conditions() and not r.passed_conditions()]
            checks_without_condition = [r for r in only_check_results if not r.have_conditions() and r.have_display()]
            checks_passed = [r for r in only_check_results if r.have_conditions() and r.passed_conditions() and
                             r.have_display()]
            checks_empty = [r.header for r in only_check_results if not r.have_display()]

            if checks_not_passed:
                display_html('<h3>Checks that didn\'t pass condition</h3>', raw=True)
                for result in checks_not_passed:
                    result._ipython_display_()
            if checks_without_condition:
                display_html('<h3>Checks without condition</h3>', raw=True)
                for result in checks_without_condition:
                    result._ipython_display_()
            if checks_passed:
                display_html('<h3>Checks that passed condition</h3>', raw=True)
                for result in checks_passed:
                    result._ipython_display_()
            if checks_empty:
                display_html('<h3>Checks with nothing found</h3>', raw=True)
                table = pd.DataFrame(data={'Check': checks_empty})
                SuiteResult._display_table(table)
        if errors_table:
            display_html('<h3>Checks that raised an error during run</h3>', raw=True)
            table = pd.DataFrame(data=errors_table, columns=['Check', 'Error'])
            SuiteResult._display_table(table)

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

    def __init__(self, name: str, *checks):
        """Get `Check`s and `CheckSuite`s to run in given order."""
        super().__init__()
        self.name = name
        self.checks = OrderedDict()

        for check in checks:
            if not isinstance(check, BaseCheck):
                raise Exception(f'CheckSuite receives only `BaseCheck` objects but got: {check.__class__.__name__}')
            if isinstance(check, CheckSuite):
                self.checks[check.name] = check
            else:
                # Check if there are already checks of the same type
                check_name = check.__class__.__name__
                same_checks = len([c for c in self.checks.values() if isinstance(c, check.__class__)])
                if same_checks:
                    check_name = f'{check_name}_{same_checks + 1}'
                self.checks[check_name] = check

    def run(self, model=None, train_dataset=None, validation_dataset=None, check_datasets_policy: str = 'validation') \
            -> SuiteResult:
        """Run all checks.

        Args:
          model: A scikit-learn-compatible fitted estimator instance
          train_dataset: Dataset object, representing data an estimator was fitted on
          validation_dataset: Dataset object, representing data an estimator predicts on
          check_datasets_policy: str, one of either ['both', 'train', 'validation'].
                                 Determines the policy by which single dataset checks are run when two datasets are
                                 given, one for train and the other for validation.

        Returns:
          List[CheckResult] - All results by all initialized checks

        Raises:
             ValueError if check_datasets_policy is not of allowed types
        """
        if check_datasets_policy not in ['both', 'train', 'validation']:
            raise ValueError('check_datasets_policy must be one of ["both", "train", "validation"]')

        # Create progress bar
        progress_bar = IntProgress(value=0, min=0, max=len(self.checks),
                                   bar_style='info', style={'bar_color': '#9d60fb'}, orientation='horizontal')
        label = HTML()
        box = VBox(children=[label, progress_bar])
        self._display_in_notebook(box)

        # Run all checks
        results = []
        for name, check in self.checks.items():
            try:
                label.value = f'Running {str(check)}'
                if isinstance(check, TrainValidationBaseCheck):
                    check_result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset,
                                             model=model)
                    check_result.set_condition_results(check.conditions_decision(check_result))
                    results.append(check_result)
                elif isinstance(check, CompareDatasetsBaseCheck):
                    check_result = check.run(dataset=validation_dataset, baseline_dataset=train_dataset, model=model)
                    check_result.set_condition_results(check.conditions_decision(check_result))
                    results.append(check_result)
                elif isinstance(check, SingleDatasetBaseCheck):
                    if check_datasets_policy in ['both', 'train'] and train_dataset is not None:
                        check_result = check.run(dataset=train_dataset, model=model)
                        check_result.header = f'{check_result.header} - Train Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                    if check_datasets_policy in ['both', 'validation'] and validation_dataset is not None:
                        check_result = check.run(dataset=validation_dataset, model=model)
                        check_result.header = f'{check_result.header} - Validation Dataset'
                        check_result.set_condition_results(check.conditions_decision(check_result))
                        results.append(check_result)
                elif isinstance(check, ModelOnlyBaseCheck):
                    check_result = check.run(model=model)
                    check_result.set_condition_results(check.conditions_decision(check_result))
                    results.append(check_result)
                elif isinstance(check, CheckSuite):
                    suite_result = check.run(model, train_dataset, validation_dataset, check_datasets_policy)
                    results.append(suite_result)
                else:
                    raise TypeError(f'Expected check of type SingleDatasetBaseCheck, CompareDatasetsBaseCheck, '
                                    f'TrainValidationBaseCheck or ModelOnlyBaseCheck. Got  {check.__class__.__name__} '
                                    f'instead')
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
        checks_str = ''.join([f'\n{c.__repr__(tabs + 1, n)}' for n, c in self.checks.items()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, item):
        """Access check inside the suite by name."""
        return self.checks[item]

    def _display_in_notebook(self, param):
        if is_notebook():
            display(param)
