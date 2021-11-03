"""Module of confidence change check."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from mlchecks import Dataset, CheckResult, TrainValidationBaseCheck
from mlchecks.checks.confidence.trust_score import TrustScore
from mlchecks.checks.confidence.preprocessing import preprocess_dataset_to_scaled_numerics
from mlchecks.metric_utils import task_type_check, ModelType
from mlchecks.utils import MLChecksValueError, model_type_validation

__all__ = ['confidence_change', 'ConfidenceChange']


def is_positive_int(x):
    return x is not None and isinstance(x, int) and x > 0


def is_float_0_to_1(x):
    return x is not None and isinstance(x, float) and 0 <= x <= 1


def validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_validation_samples):
    if not is_positive_int(k_filter):
        raise MLChecksValueError(f'k_filter must be positive integer but got: {k_filter}')
    if not is_float_0_to_1(alpha):
        raise MLChecksValueError(f'alpha must be float between 0 to 1 but got: {alpha}')
    if not is_float_0_to_1(bin_size):
        raise MLChecksValueError(f'bin_size must be float between 0 to 1 but got: {bin_size}')
    if not is_positive_int(max_number_categories):
        raise MLChecksValueError(f'max_number_categories must be positive integer but got: {max_number_categories}')
    if not is_positive_int(min_validation_samples):
        raise MLChecksValueError(f'min_validation_samples must be positive integer but got: {min_validation_samples}')


def confidence_change(train_dataset, validation_dataset, model,
                      k_filter: int = 10, alpha: float = 0.001, bin_size: float = 0.02,
                      max_number_categories: int = 10, min_validation_samples: int = 300):
    """Check whether the confidence of the model changed, in order to understand if important features drifted.

    The process is as follows:
    * Pre-process the train and validation data into scaled numerics
    * Train a TrustScore (https://arxiv.org/abs/1805.11783) regressor based on train data + label.
    * Project the TrustScore scores on train data + label to uniform distribution (binning into percentiles)
    * Use TrustScore to score the prediction of the model, and project to the same uniform distribution.
    * The mean of the above distribution should be 0.5 if data is identical to train data. In practice, even data that
    belongs to the train data but wasn't trained on, is a bit skewed, so preferably we would use a validation dataset
    which we know isn't skewed (NOT IMPLEMENTED NOW).
    * The check measures the distance between the baseline mean of the distribution (0.5) to the observed, and projects
    it to [-1, 1]. A score around 0 means no drift in confidence, a score around 1 means drift to the worse, and a score
    around -1 means a drift that improves results (usually will not happen).
    * Currently, as we don't compare to the validation data, the result will rarely be around 0. Therefore, it is
    advised to measure the CHANGE in this metric, and not the absolute score.

    Args:
        train_dataset (Dataset): Dataset to use for TrustScore regressor
        validation_dataset (Dataset): Dataset to check for confidence
        model: Model used to predict on the validation dataset
        k_filter (int): used in TrustScore (Number of neighbors used during either kNN distance or probability
                        filtering)
        alpha (float): used in TrustScore (Fraction of instances to filter out to reduce impact of outliers)
        bin_size (float): Number of percentiles of the confidence train to fit on
        max_number_categories (int): Indicates the maximum number of unique categories in a single categorical column
                                    (rare categories will be changed to a form of "other")
        min_validation_samples (int): Minimal number of samples in train data to be able to run this check
    """
    self = confidence_change
    # Validations
    validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_validation_samples)
    # tested dataset can be also dataframe
    validation_dataset: Dataset = Dataset.validate_dataset_or_dataframe(validation_dataset)
    model_type_validation(model)
    model_type = task_type_check(model, validation_dataset)
    validation_dataset.validate_model(model)
    # Baseline must have label so we must get it as Dataset.
    Dataset.validate_dataset(train_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    train_dataset.validate_shared_features(validation_dataset, self.__name__)

    if validation_dataset.n_samples() < min_validation_samples:
        msg = ('<i>Did not run since number of samples in validation have not passed the minimum. you can change '
               'minimum samples needed to run with parameter "min_validation_samples"</i>')
        return CheckResult(None, check=self, display=msg)
    if model_type == ModelType.REGRESSION:
        raise MLChecksValueError(f'Regression models are not supported for {self.__name__}')

    train_dataset = train_dataset.sample(10000)
    validation_dataset = validation_dataset.sample(10000)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_dataset.label_col())

    x_baseline, x_tested = preprocess_dataset_to_scaled_numerics(baseline_features=train_dataset.features_columns(),
                                                                 test_features=validation_dataset.features_columns(),
                                                                 categorical_columns=validation_dataset.cat_features(),
                                                                 max_num_categories=max_number_categories)

    y_baseline = label_encoder.transform(train_dataset.label_col())
    trust_score_model = TrustScore(k_filter=k_filter, alpha=alpha)
    trust_score_model.fit(X=x_baseline.to_numpy(), Y=y_baseline)
    baseline_confidences = trust_score_model.score(x_baseline.to_numpy(), y_baseline)[0].astype('float64')
    bin_range = np.arange(0, 1, bin_size)
    fitted_bins = np.quantile(a=baseline_confidences, q=bin_range)
    # Calculate y on tested dataset using the model
    y_tested = model.predict(validation_dataset.features_columns())
    y_tested_encoded = label_encoder.transform(y_tested)
    tested_confidences = trust_score_model.score(x_tested.to_numpy(), y_tested_encoded)[0].astype('float64')
    transposed_tested_confidences = np.digitize(tested_confidences, fitted_bins) * bin_size

    # Add confidence and prediction and sort by confidence
    x_tested.insert(0, 'Model Prediction', y_tested)
    x_tested.insert(0, 'Confidence Quantile', transposed_tested_confidences)
    x_tested = x_tested.sort_values(by=['Confidence Quantile'], ascending=False)
    # x_tested = x_tested.set_index(['confidence', 'y'])
    columns_to_show = ['Model Prediction', 'Confidence Quantile']
    if validation_dataset.index_name() is not None:
        columns_to_show.append(validation_dataset.index_name())

    # Display top 5 and bottom 5
    k = 5
    top_k = x_tested.head(k)[columns_to_show]
    bottom_k = x_tested.tail(k)[columns_to_show]
    tested_confidence_mean = np.mean(transposed_tested_confidences)

    def display_plot():
        # The fitted bins are allocated to the right side of the range so the first bin is `bin_size` and not 0,
        # and the last bin is 1.
        display_bins = np.arange(bin_size, 1 + bin_size, bin_size)
        s_bins = pd.Series(display_bins, name='bins', index=display_bins)
        # Tested dataset
        tested_confidences_value_counts = (pd.Series(transposed_tested_confidences).value_counts().rename('values')
                                           / transposed_tested_confidences.sum())
        dataset_distribution = pd.DataFrame(s_bins, index=display_bins).join(tested_confidences_value_counts).fillna(0)

        # Plotting the quantile at the middle of the bin
        dataset_distribution['bins'] = dataset_distribution['bins'] - bin_size / 2
        # Baseline
        # transposed_baseline_confidences = np.digitize(baseline_confidences, fitted_bins, right=True) * bin_size
        # baseline_confidences_value_counts = (pd.Series(transposed_baseline_confidences).value_counts()
        #                                       .rename('values') / transposed_baseline_confidences.sum())
        # baseline_distribution = pd.DataFrame(s_bins, index=bin_range)\
        #                           .join(baseline_confidences_value_counts).fillna(0)
        #
        # plt.bar(baseline_distribution['bins'], height=baseline_distribution['values'], color='purple',
        #         width=bin_size, alpha=0.7)
        _, axes = plt.subplots(1, 1, figsize=(7, 4))
        axes.set_xlim([0, 1])
        axes.set_ylim([0, max(dataset_distribution['values']) + 0.01])
        plt.bar(dataset_distribution['bins'], dataset_distribution['values'], color='darkblue', width=bin_size)
        plt.plot([0, 1], [bin_size, bin_size], color='purple', lw=3)
        colors = {'Validation confidence distribution normalized by train distribution': 'darkblue',
                  'Uniform train confidence distribution': 'purple'}
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        plt.legend(handles, labels)
        plt.ylabel('Density')
        plt.xlabel('Confidence Quantile')
        plt.title('Distribution of samples confidence per confidence quantile')
        plt.show()

    display = [display_plot, '<h5>Top Confidence Samples Indexes</h5>', top_k,
               '<h5>Worst Confidence Samples Indexes</h5>', bottom_k]
    return CheckResult(tested_confidence_mean, check=self, display=display)


class ConfidenceChange(TrainValidationBaseCheck):
    """Checks whether the confidence of the model changed, in order to understand if important features drifted.

    The process is as follows:
    * Pre-process the train and validation data into scaled numerics
    * Train a TrustScore (https://arxiv.org/abs/1805.11783) regressor based on train data + label.
    * Project the TrustScore scores on train data + label to uniform distribution (binning into percentiles)
    * Use TrustScore to score the prediction of the model, and project to the same uniform distribution.
    * The mean of the above distribution should be 0.5 if data is identical to train data. In practice, even data that
    belongs to the train data but wasn't trained on, is a bit skewed, so preferably we would use a validation dataset
    which we know isn't skewed (NOT IMPLEMENTED NOW).
    * The check measures the distance between the baseline mean of the distribution (0.5) to the observed, and projects
    it to [-1, 1]. A score around 0 means no drift in confidence, a score around 1 means drift to the worse, and a score
    around -1 means a drift that improves results (usually will not happen).
    * Currently, as we don't compare to the validation data, the result will rarely be around 0. Therefore, it is
    advised to measure the CHANGE in this metric, and not the absolute score.

    Parameters:
        k_filter (int): used in TrustScore (Number of neighbors used during either kNN distance or probability
                        filtering)
        alpha (float): used in TrustScore (Fraction of instances to filter out to reduce impact of outliers)
        bin_size (float): Number of percentiles of the confidence train to fit on
        max_number_categories (int): Indicates the maximum number of unique categories in a single categorical column
                                    (rare categories will be changed to a form of "other")
        min_validation_samples (int): Minimal number of samples in train data to be able to run this check
    """

    def run(self, train_dataset, validation_dataset, model=None) -> CheckResult:
        """Check whether the confidence of the model changed, in order to understand if important features drifted.

        Args:
            train_dataset (Dataset): Dataset to use for TrustScore regressor
            validation_dataset (Dataset): Dataset to check for confidence
            model: Model used to predict on the validation dataset
        """
        return confidence_change(train_dataset, validation_dataset, model, **self.params)
