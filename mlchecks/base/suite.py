"""Module containing the Suite object, used for running a set of checks together."""
import typing as t

from IPython.core.display import display_html, display
from ipywidgets import IntProgress, HTML, VBox

from mlchecks import base
from mlchecks import utils


__all__ = ["CheckSuite"]


CheckPolicy = t.Union[t.Literal["both"], t.Literal["train"], t.Literal["validation"]]


class CheckSuite(base.BaseCheck):
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Attributes:
        checks: A list of checks to run.
    """

    checks: t.List[base.BaseCheck]
    name: str

    def __init__(self, name: str, *checks: base.BaseCheck):
        """Get `Check`s and `CheckSuite`s to run in given order."""
        super().__init__()
        
        self.name = name
        self.checks = []

        for c in checks:
            if not isinstance(c, base.BaseCheck):
                raise TypeError(f"CheckSuite receives only `BaseCheck` objects but got: {type(c)}")
            if isinstance(c, CheckSuite):
                self.checks.extend(c.checks)
            else:
                self.checks.append(c)

    def run(
        self, 
        train_dataset: t.Optional[base.Dataset] = None, 
        validation_dataset: t.Optional[base.Dataset] = None, 
        model: object = None, 
        check_datasets_policy: CheckPolicy = "validation"
    ) -> base.CheckResult:
        """Run all checks.

        Args:
            model: A scikit-learn-compatible fitted estimator instance
            train_dataset: Dataset object, representing data an estimator was fitted on
            validation_dataset: Dataset object, representing data an estimator predicts on
            check_datasets_policy: Union[Literal["both"], Literal["train"], Literal["validation"]],
                                    Determines the policy by which single dataset checks are run when two datasets are
                                    given, one for train and the other for validation.

        Returns:
            List[CheckResult] - All results by all initialized checks

        Raises:
            TypeError if check_datasets_policy is not of allowed types
        """
        if check_datasets_policy not in ["both", "train", "validation"]:
            raise TypeError(f"check_datasets_policy must be one of {repr(CheckPolicy)}")

        # Create progress bar
        progress_bar = IntProgress(value=0, min=0, max=len(self.checks),
                                   bar_style="info", style={"bar_color": "#9d60fb"}, orientation="horizontal")
        
        label = HTML()
        box = VBox(children=[label, progress_bar])
        self._display_in_notebook(box)

        # Run all checks
        results = []
        
        for check in self.checks:
            label.value = f"Running {str(check)}"

            if train_dataset is not None and validation_dataset is not None:
                if isinstance(check, base.TrainValidationBaseCheck):
                    results.append(check.run(
                        train_dataset=train_dataset, 
                        validation_dataset=validation_dataset, 
                        model=model
                    ))
                elif isinstance(check, base.CompareDatasetsBaseCheck):
                    results.append(check.run(
                        dataset=validation_dataset, 
                        baseline_dataset=train_dataset, 
                        model=model
                    ))
            
            elif isinstance(check, base.SingleDatasetBaseCheck):
                if check_datasets_policy in {"both", "train"} and train_dataset is not None:
                    check_result = check.run(dataset=train_dataset, model=model)
                    check_result.header = f"{check_result.header} - Train Dataset"
                    results.append(check_result)
                if check_datasets_policy in {"both", "validation"} and validation_dataset is not None:
                    check_result = check.run(dataset=validation_dataset, model=model)
                    check_result.header = f"{check_result.header} - Validation Dataset"
                    results.append(check_result)
            
            elif isinstance(check, base.ModelOnlyBaseCheck):
                results.append(check.run(model=model))
            
            else:
                raise TypeError(
                    "Expected check of type SingleDatasetBaseCheck, CompareDatasetsBaseCheck, "
                    f"TrainValidationBaseCheck or ModelOnlyBaseCheck. Got {check.__class__.__name__} "
                    "instead"
                )
            
            progress_bar.value = progress_bar.value + 1

        progress_bar.close()
        label.close()
        box.close()

        def display_suite():
            display_html(f"<h3>{self.name}</h3>", raw=True)
            for result in results:
                # Disable protected access warning
                #pylint: disable=protected-access
                result._ipython_display_()

        return base.CheckResult(results, display=display_suite)

    def __repr__(self):
        checks_str = ",".join([str(c) for c in self.checks])
        return f"{self.name} [{checks_str}]"

    def _display_in_notebook(self, param):
        if utils.is_notebook():
            display(param)
