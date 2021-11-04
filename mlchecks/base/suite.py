"""Module containing the Suite object, used for running a set of checks together."""
from collections import OrderedDict
from typing import Union, List

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

    def _ipython_display_(self, verbose=False):
        display_html(f'<h3>{self.name}</h3>', raw=True)
        # First print summary
        display_table = []
        for result in self.results:
            if isinstance(result, CheckResult):
                for cond_name, cond_result in result.conditions_results.items():
                    display_table.append([cond_result.is_pass, result.header, cond_name, cond_result.actual,
                                          cond_result.category])
        table = pd.DataFrame(data=display_table, columns=['Pass', 'Check', 'Condition', 'Actual', 'Category'])
        df_styler = table.style
        df_styler.set_table_styles([dict(selector='th,td', props=[('text-align', 'left')])])
        display_html(df_styler.render(), raw=True)
        # If verbose print all displays
        if verbose:
            for result in self.results:
                # pylint: disable=protected-access
                result._ipython_display_()

    def display_verbose(self):
        """Display the suite result with verbose output of each check."""
        self._ipython_display_(verbose=True)


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
                self.checks[check.__class__.__name__] = check

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
        for check in self.checks.values():
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
            progress_bar.value = progress_bar.value + 1

        progress_bar.close()
        label.close()
        box.close()

        return SuiteResult(self.name, results)

    def __repr__(self, tabs=0):
        """Representation of suite as string."""
        tabs_str = '\t' * tabs
        checks_str = ''.join([f'\n{c.__repr__(tabs + 1)}' for c in self.checks.values()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, item):
        """Access check inside the suite by name."""
        return self.checks[item]

    def _display_in_notebook(self, param):
        if is_notebook():
            display(param)
