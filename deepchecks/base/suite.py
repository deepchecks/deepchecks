"""Module containing the Suite object, used for running a set of checks together."""
# pylint: disable=broad-except
from collections import OrderedDict
from typing import Union, List

from IPython.core.display import display
from ipywidgets import IntProgress, HTML, VBox

from deepchecks.base.check import BaseCheck, CheckResult, TrainTestBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck, CheckFailure

__all__ = ['CheckSuite', 'SuiteResult']

from deepchecks.base.display_suite import display_suite_result_2

from deepchecks.utils import DeepchecksValueError, is_widgets_enabled


class SuiteResult:
    """Contain the results of a suite run."""

    name: str
    results: List[Union[CheckResult, CheckFailure, str]]

    def __init__(self, name: str, results):
        """Initialize suite result."""
        self.name = name
        self.results = results

    def _ipython_display_(self):
        display_suite_result_2(self.name, self.results)


class CheckSuite(BaseCheck):
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
    """

    checks: OrderedDict
    name: str
    _check_index: int

    def __init__(self, name: str, *checks: Union[str, BaseCheck]):
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
        for check in self.checks.values():
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
                elif isinstance(check, str):
                    results.append(check)
                else:
                    raise TypeError(f'Don\'t know how to handle type {check.__class__.__name__} in suite.')
            except Exception as exp:
                results.append(CheckFailure(check.__class__, exp))
            progress_bar.value = progress_bar.value + 1

        progress_bar.close()
        label.close()
        box.close()

        return SuiteResult(self.name, results)

    def __repr__(self, tabs=0):
        """Representation of suite as string."""
        additional_info = []
        stringified_checks = []

        for index, check in self.checks.items():
            if isinstance(check, str):
                indent = '\t' * (tabs + 2)
                additional_info.append(f'\n{indent}- {check}')
            else:
                check_repr = check.__repr__(tabs + 1, str(index) + ': ')
                stringified_checks.append(f'\n{check_repr}')

        aditional_info_indent = '\t' * (tabs + 1)
        info = ''.join(additional_info)
        aditional_info_section = f'{aditional_info_indent}Additional Info:{info}'

        indent = '\t' * tabs
        checks = ''.join(stringified_checks)

        return (
            f'{indent}{self.name}: [{checks}\n\n{aditional_info_section}\n{indent}]'
        )

    def __getitem__(self, index) -> Union[str, BaseCheck]:
        """Access check inside the suite by name."""
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        return self.checks[index]

    def add(self, check: Union[str, BaseCheck]) -> 'CheckSuite':
        """Add a check or a suite to current suite.

        Args:
            check (BaseCheck): A check or suite to add.
        """
        if not isinstance(check, (BaseCheck, str)):
            raise Exception(
                f"'{type(self).__name__}' receives only `{BaseCheck.__name__}` or string "
                f"objects but got: '{type(check).__name__}'"
            )
        if isinstance(check, CheckSuite):
            for c in check.checks.values():
                self.add(c)
        else:
            self.checks[self._check_index] = check
            self._check_index += 1
        return self

    def remove(self, index: int) -> 'CheckSuite':
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
