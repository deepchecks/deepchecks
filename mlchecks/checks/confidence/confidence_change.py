"""Module of confidence change check."""
import numpy as np
import pandas as pd
from mlchecks import Dataset, CheckResult
from mlchecks.checks.confidence.measures import TrustScore
from mlchecks.checks.confidence.preprocessing import preprocess_dataset_to_scaled_numerics
from mlchecks.metric_utils import task_type_check, ModelType
from mlchecks.utils import MLChecksValueError, model_type_validation

__all__ = ['confidence_change']


def is_positive_int(x):
    return x is not None and isinstance(x, int) and x > 0


def is_float_0_to_1(x):
    return x is not None and isinstance(x, float) and 0 <= x <= 1


def validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_baseline_samples):
    if not is_positive_int(k_filter):
        raise MLChecksValueError(f'k_filter must be positive integer but got: {k_filter}')
    if not is_float_0_to_1(alpha):
        raise MLChecksValueError(f'alpha must be float between 0 to 1 but got: {alpha}')
    if not is_float_0_to_1(bin_size):
        raise MLChecksValueError(f'bin_size must be float between 0 to 1 but got: {bin_size}')
    if not is_positive_int(max_number_categories):
        raise MLChecksValueError(f'max_number_categories must be positive integer but got: {max_number_categories}')
    if not is_positive_int(min_baseline_samples):
        raise MLChecksValueError(f'min_baseline_samples must be positive integer but got: {min_baseline_samples}')


def confidence_change(dataset, baseline_dataset, model,
                      k_filter: int = 10, alpha: float = 0.001, bin_size: float = 0.01,
                      max_number_categories: int = 10, min_baseline_samples: int = 300):
    """Confidence"""
    self = confidence_change
    # Validations
    validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_baseline_samples)
    # tested dataset can be also dataframe
    dataset: Dataset = Dataset.validate_dataset_or_dataframe(dataset)
    model_type_validation(model)
    model_type = task_type_check(model, dataset)
    dataset.validate_model(model)
    # Baseline must have label so we must get it as Dataset.
    baseline_dataset: Dataset = Dataset.validate_dataset(baseline_dataset, self.__name__)
    baseline_dataset.validate_label(self.__name__)
    baseline_dataset.validate_shared_features(dataset, self.__name__)

    if baseline_dataset.n_samples() < min_baseline_samples or model_type != ModelType.BINARY:
        return CheckResult(None, check=self)

    x_baseline, x_tested = preprocess_dataset_to_scaled_numerics(baseline_features=baseline_dataset.features_columns(),
                                                                 test_features=dataset.features_columns(),
                                                                 categorical_columns=dataset.cat_features(),
                                                                 max_num_categories=max_number_categories)

    y_baseline = baseline_dataset.label_col()
    num_classes = y_baseline.nunique()
    trust_score_model = TrustScore(k_filter=k_filter, alpha=alpha)
    trust_score_model.fit(X=x_baseline.to_numpy(), Y=y_baseline.to_numpy(), classes=num_classes)
    baseline_confidences = trust_score_model.score(x_baseline.to_numpy(), y_baseline.to_numpy())[0].astype('float64')

    bin_range = np.arange(0, 1, bin_size)
    fitted_bins = np.quantile(a=baseline_confidences, q=bin_range)

    # Calculate y on tested dataset using the model
    y_tested = model.predict(dataset.features_columns())
    tested_confidences = trust_score_model.score(x_tested.to_numpy(), y_tested)[0].astype('float64')

    transposed_tested_confidences = np.digitize(tested_confidences, fitted_bins, right=True) * bin_size
    result = np.round(0.5 - np.mean(transposed_tested_confidences), 2) * 2
    s_bins = pd.Series(bin_range, name='bins', index=bin_range)

    # Tested dataset
    tested_confidences_value_counts = (pd.Series(transposed_tested_confidences).value_counts().rename('values')
                                       / transposed_tested_confidences.sum())
    dataset_distribution = pd.DataFrame(s_bins, index=bin_range).join(tested_confidences_value_counts).fillna(0)

    # Baseline
    transposed_baseline_confidences = np.digitize(baseline_confidences, fitted_bins, right=True) * bin_size
    baseline_confidences_value_counts = (pd.Series(transposed_baseline_confidences).value_counts().rename('values')
                                         / transposed_baseline_confidences.sum())
    baseline_distribution = pd.DataFrame(s_bins, index=bin_range).join(baseline_confidences_value_counts).fillna(0)

    result = {
        'tested_confidence_mean': np.mean(transposed_tested_confidences),
        'baseline_confidence_mean': np.mean(transposed_baseline_confidences),
        'tested_confidence_distribution': dataset_distribution,
        'baseline_confidence_distribution': baseline_distribution,
    }

    dataframe_mean = pd.DataFrame(data={'Confidence Mean': [result['tested_confidence_mean']],
                                        'Baseline  Confidence Mean': [result['baseline_confidence_mean']]})

    display = [dataframe_mean, dataset_distribution, baseline_distribution]
    return CheckResult(result, check=self, display=display)
