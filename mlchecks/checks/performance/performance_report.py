"""The classification_report check module."""
from typing import Callable, Dict

import pandas as pd
from sklearn.base import BaseEstimator
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck


__all__ = ['classification_report', 'ClassificationReport']

from mlchecks.metric_utils import get_metrics_list

from mlchecks.utils import model_type_validation


def classification_report(dataset: Dataset, model, alternative_metrics: Dict[str, Callable] = None):
    """Summarize given metrics on a dataset and model.

    Args:
        dataset (Dataset): a Dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions. If none
        given, using default metrics

    Returns:
        CheckResult: value is dictionary in format `{metric: score, ...}`
    """
    self = classification_report
    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)
    model_type_validation(model)

    # Get default metrics if no alternative, or validate alternatives
    metrics = get_metrics_list(model, dataset, alternative_metrics)
    scores = {key: scorer(model, dataset.features_columns(), dataset.label_col()) for key, scorer in metrics.items()}

    display_df = pd.DataFrame(data=[*scores.items()], columns=['Metric', 'Score'])
    display_df.set_index('Metric')

    return CheckResult(scores, check=self, display=display_df)


class ClassificationReport(SingleDatasetBaseCheck):
    """Summarize given metrics on a dataset and model."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run classification_report check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format `{<metric>: score}`
        """
        return classification_report(dataset, model, **self.params)
