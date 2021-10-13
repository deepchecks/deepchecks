from typing import List

from mlchecks.utils import model_type_validation, MLChecksValueError
from mlchecks import validate_dataset, Dataset, CheckResult

DEFAULT_BINARY_METRICS = ['auc']
DEFAULT_MULTICLASS_METRICS = ['auc']
DEFAULT_REGRESSION_METRICS = ['rmse']

def auc(model, dataset: Dataset):
    pass


def _multiclass_classification(train_dataset: Dataset, validation_dataset: Dataset, model, metrics: List[str]):
    for metric in metrics:
        pass


def _binary_classification(train_dataset: Dataset, validation_dataset: Dataset, model, metrics: List[str]):
    for metric in metrics:
        pass


def _regression(train_dataset: Dataset, validation_dataset: Dataset, model, metrics: List[str]):
    pass


def _validate_metrics(additional_metrics, default_metrics, supported_metrics):
    if additional_metrics is None:
        return default_metrics
    elif isinstance(additional_metrics, List):
        # Check all values inside list are strings
        if any(not isinstance(x, str) for x in additional_metrics):
            raise MLChecksValueError(f'additional_metrics list must contain only strings of: '
                                     f'{", ".join(supported_metrics)}')
        # Check got metrics we support
        unsupported = set(additional_metrics) - set(supported_metrics)
        if unsupported:
            raise MLChecksValueError(f'Unsupported metrics: {", ".join(unsupported)}')
    else:
        raise MLChecksValueError(f'additional_metrics type must be `List[str]` but got: '
                                 f'{type(additional_metrics).__name__}')


def performance_overfit(train_dataset: Dataset, validation_dataset: Dataset, model,
                        additional_metrics: List[str] = None) -> CheckResult:
    """Performance overfit.

    Args:
        train_dataset
        validation_dataset
        model
        additional_metrics
    """
    # Validate parameters
    func_name = 'performance_overfit'
    model_type_validation(model)
    validate_dataset(train_dataset, func_name)
    validate_dataset(validation_dataset, func_name)
    train_dataset.validate_shared_label(validation_dataset, func_name)
    train_dataset.validate_shared_features(validation_dataset, func_name)

    # Check for model type
    if 'predict_proba' in model.__dir__:
        # Check if binary or multiclass
        if train_dataset.label_col().nunique() <= 2:
            metrics = _validate_metrics(additional_metrics, DEFAULT_BINARY_METRICS, [])
            _binary_classification(train_dataset, validation_dataset, model, metrics)
        else:
            metrics = _validate_metrics(additional_metrics, DEFAULT_MULTICLASS_METRICS, [])
            _multiclass_classification(train_dataset, validation_dataset, model, metrics)
    else:
        metrics = _validate_metrics(additional_metrics, DEFAULT_REGRESSION_METRICS, [])
        _regression(train_dataset, validation_dataset, model, metrics)
