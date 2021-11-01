"""Module containing the Suite object, used for running a set of checks together."""
from collections import OrderedDict

from IPython.core.display import display_html, display
from ipywidgets import IntProgress, HTML, VBox

from mlchecks.base.check import BaseCheck, CheckResult, TrainValidationBaseCheck, CompareDatasetsBaseCheck, \
    SingleDatasetBaseCheck, ModelOnlyBaseCheck

__all__ = ['CheckSuite']

from mlchecks.utils import MLChecksValueError, is_notebook


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
            self.checks[check.__class__.__name__] = check

    def run(self, model=None, train_dataset=None, validation_dataset=None, check_datasets_policy: str = 'validation') \
            -> CheckResult:
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
                results.append(check.run(train_dataset=train_dataset, validation_dataset=validation_dataset,
                                         model=model))
            elif isinstance(check, CompareDatasetsBaseCheck):
                results.append(check.run(dataset=validation_dataset, baseline_dataset=train_dataset, model=model))
            elif isinstance(check, SingleDatasetBaseCheck):
                if check_datasets_policy in ['both', 'train'] and train_dataset is not None:
                    res = check.run(dataset=train_dataset, model=model)
                    res.header = f'{res.header} - Train Dataset'
                    results.append(res)
                if check_datasets_policy in ['both', 'validation'] and validation_dataset is not None:
                    res = check.run(dataset=validation_dataset, model=model)
                    res.header = f'{res.header} - Validation Dataset'
                    results.append(res)
            elif isinstance(check, ModelOnlyBaseCheck):
                results.append(check.run(model=model))
            elif isinstance(check, CheckSuite):
                suite_res = check.run(model, train_dataset, validation_dataset, check_datasets_policy)
                if check.name in results:
                    raise MLChecksValueError('Each suite must have a unique name')
                results.append(suite_res)
            else:
                raise TypeError(f'Expected check of type SingleDatasetBaseCheck, CompareDatasetsBaseCheck, '
                                f'TrainValidationBaseCheck or ModelOnlyBaseCheck. Got  {check.__class__.__name__} '
                                f'instead')
            progress_bar.value = progress_bar.value + 1

        progress_bar.close()
        label.close()
        box.close()

        def display_suite():
            display_html(f'<h3>{self.name}</h3>', raw=True)
            for result in results:
                # Disable protected access warning
                #pylint: disable=protected-access
                result._ipython_display_()

        return CheckResult(results, display=display_suite)

    def __repr__(self, tabs=0):
        """Representation of suite as string."""
        tabs_str = '\t' * tabs
        checks_str = ''.join([f'\n{c.__repr__(tabs + 1)}' for c in self.checks.values()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, item):
        return self.checks[item]

    def _display_in_notebook(self, param):
        if is_notebook():
            display(param)
