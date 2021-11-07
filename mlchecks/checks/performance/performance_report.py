"""Module containing performance report check."""
from typing import Callable, Dict
import pandas as pd
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from mlchecks.metric_utils import get_metrics_list
from mlchecks.utils import model_type_validation

__all__ = ['PerformanceReport']


class PerformanceReport(SingleDatasetBaseCheck):
    """Summarize given metrics on a dataset and model."""

    def __init__(self, alternative_metrics: Dict[str, Callable] = None, **params):
        """Initialize the PerformanceReport check.

        Args:
            alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions.
            If none given, using default metrics
        """
        super().__init__(**params)
        self.alternative_metrics = alternative_metrics

    def run(self, dataset, model=None) -> CheckResult:
        """Summarize given metrics on a dataset and model.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format `{<metric>: score}`
        """
        return self._performance_report(dataset, model)

    def _performance_report(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset, self._performance_report.__name__)
        dataset.validate_label(self._performance_report.__name__)
        model_type_validation(model)

        # Get default metrics if no alternative, or validate alternatives
        metrics = get_metrics_list(model, dataset, self.alternative_metrics)
        scores = {key: scorer(model, dataset.features_columns(), dataset.label_col()) for key, scorer in
                  metrics.items()}

        display_df = pd.DataFrame(scores.values(), columns=['Score'], index=scores.keys())
        display_df.index.name = 'Metric'

        return CheckResult(scores, check=self.run, header='Performance Report', display=display_df)
