"""Module of confidence change check."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck
from deepchecks.checks.confidence.trust_score import TrustScore
from deepchecks.checks.confidence.preprocessing import preprocess_dataset_to_scaled_numerics
from deepchecks.metric_utils import task_type_check, ModelType
from deepchecks.utils import DeepchecksValueError, model_type_validation

__all__ = ['ConfidenceChange']


def is_positive_int(x):
    return x is not None and isinstance(x, int) and x > 0


def is_float_0_to_1(x):
    return x is not None and isinstance(x, float) and 0 <= x <= 1


def validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_test_samples, sample_size, n_to_show):
    if not is_positive_int(k_filter):
        raise DeepchecksValueError(f'k_filter must be positive integer but got: {k_filter}')
    if not is_float_0_to_1(alpha):
        raise DeepchecksValueError(f'alpha must be float between 0 to 1 but got: {alpha}')
    if not is_float_0_to_1(bin_size):
        raise DeepchecksValueError(f'bin_size must be float between 0 to 1 but got: {bin_size}')
    if not is_positive_int(max_number_categories):
        raise DeepchecksValueError(f'max_number_categories must be positive integer but got: {max_number_categories}')
    if not is_positive_int(min_test_samples):
        raise DeepchecksValueError(f'min_test_samples must be positive integer but got: {min_test_samples}')
    if not is_positive_int(sample_size):
        raise DeepchecksValueError(f'sample_size must be positive integer but got: {min_test_samples}')
    if sample_size < min_test_samples:
        raise DeepchecksValueError(f'sample_size can\'t be smaller than min_test_samples')
    if not is_positive_int(n_to_show):
        raise DeepchecksValueError(f'n_to_show must be positive integer but got: {min_test_samples}')


class ConfidenceChange(TrainTestBaseCheck):
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
    """

    def __init__(self, k_filter: int = 10, alpha: float = 0.001, bin_size: float = 0.02,
                 max_number_categories: int = 10, min_test_samples: int = 300, sample_size: int = 10_000,
                 random_state: int = 42, n_to_show: int = 5):
        """
        Args:
            k_filter (int): used in TrustScore (Number of neighbors used during either kNN distance or probability
                            filtering)
            alpha (float): used in TrustScore (Fraction of instances to filter out to reduce impact of outliers)
            bin_size (float): Number of percentiles of the confidence train to fit on
            max_number_categories (int): Indicates the maximum number of unique categories in a single categorical
                                         column (rare categories will be changed to a form of "other")
            min_test_samples (int): Minimal number of samples in train data to be able to run this check
            sample_size (int): Number of samples to use for the check for train and test. if dataset contains less than
                               sample_size than all the dataset will be used.
            random_state (int): The random state to use for sampling.
            n_to_show (int): Number of samples to show of worst and best confidence.
        """
        validate_parameters(k_filter, alpha, bin_size, max_number_categories, min_test_samples, sample_size, n_to_show)
        self.k_filter = k_filter
        self.alpha = alpha
        self.bin_size = bin_size
        self.max_number_categories = max_number_categories
        self.min_test_samples = min_test_samples
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_to_show = n_to_show

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Check whether the confidence of the model changed, in order to understand if important features drifted.

        Args:
            train_dataset (Dataset): Dataset to use for TrustScore regressor
            test_dataset (Dataset): Dataset to check for confidence
            model: Model used to predict on the validation dataset
        """
        # tested dataset can be also dataframe
        test_dataset: Dataset = Dataset.validate_dataset_or_dataframe(test_dataset)
        model_type_validation(model)
        model_type = task_type_check(model, test_dataset)
        test_dataset.validate_model(model)
        # Baseline must have label so we must get it as Dataset.
        Dataset.validate_dataset(train_dataset, self.__class__.__name__)
        train_dataset.validate_label(self.__class__.__name__)
        train_dataset.validate_shared_features(test_dataset, self.__class__.__name__)

        if test_dataset.n_samples() < self.min_test_samples:
            msg = ('Number of samples in test dataset have not passed the minimum. you can change '
                   'minimum samples needed to run with parameter "min_test_samples"')
            raise DeepchecksValueError(msg)
        if model_type != ModelType.BINARY:
            raise DeepchecksValueError(f'Check supports only binary classification')

        train_data_sample = train_dataset.data.sample(min(self.sample_size, train_dataset.n_samples()),
                                                      random_state=self.random_state)
        test_data_sample = test_dataset.data.sample(min(self.sample_size, test_dataset.n_samples()),
                                                    random_state=self.random_state)
        features_list = train_dataset.features()
        label_name = train_dataset.label_name()

        x_train, x_test = preprocess_dataset_to_scaled_numerics(
            baseline_features=train_data_sample[features_list],
            test_features=test_data_sample[features_list],
            categorical_columns=test_dataset.cat_features,
            max_num_categories=self.max_number_categories
        )

        y_train = train_data_sample[label_name]
        trust_score_model = TrustScore(k_filter=self.k_filter, alpha=self.alpha)
        trust_score_model.fit(X=x_train.to_numpy(), Y=y_train)
        train_confidences = trust_score_model.score(x_train.to_numpy(), y_train)[0].astype('float64')
        bin_range = np.arange(0, 1, self.bin_size)
        fitted_bins = np.quantile(a=train_confidences, q=bin_range)
        # Calculate y on tested dataset using the model
        y_test = model.predict(test_data_sample[features_list])
        tested_confidences = trust_score_model.score(x_test.to_numpy(), y_test)[0].astype('float64')
        transposed_tested_confidences = np.digitize(tested_confidences, fitted_bins) * self.bin_size

        # Add confidence and prediction and sort by confidence
        x_test.insert(0, 'Model Prediction', y_test)
        if test_dataset.label_name():
            x_test.insert(0, test_dataset.label_name(), test_data_sample[test_dataset.label_name()])
        if test_dataset.index_name():
            x_test.insert(0, test_dataset.index_name(), test_data_sample[test_dataset.index_name()])
        x_test.insert(0, 'Confidence Quantile', transposed_tested_confidences)
        x_test = x_test.sort_values(by=['Confidence Quantile'], ascending=False)

        # Display top and bottom
        top_k = x_test.head(self.n_to_show)
        bottom_k = x_test.tail(self.n_to_show)
        tested_confidence_mean = np.mean(transposed_tested_confidences)

        print(tested_confidences)

        def display_plot(bin_size=self.bin_size):
            # The fitted bins are allocated to the right side of the range so the first bin is `bin_size` and not 0,
            # and the last bin is 1.
            display_bins = np.arange(bin_size, 1 + bin_size, bin_size)
            s_bins = pd.Series(display_bins, name='bins', index=display_bins)
            # Tested dataset
            tested_confidences_value_counts = (pd.Series(transposed_tested_confidences).value_counts().rename('values')
                                               / transposed_tested_confidences.sum())
            dataset_distribution = pd.DataFrame(s_bins, index=display_bins).join(
                tested_confidences_value_counts).fillna(0)

            # Plotting the quantile at the middle of the bin
            dataset_distribution['bins'] = dataset_distribution['bins'] - bin_size / 2
            _, axes = plt.subplots(1, 1, figsize=(7, 4))
            axes.set_xlim([0, 1])
            axes.set_ylim([0, max(dataset_distribution['values']) + 0.01])
            plt.bar(dataset_distribution['bins'], dataset_distribution['values'], color='darkblue', width=bin_size)
            plt.plot([0, 1], [bin_size, bin_size], color='purple', lw=3)
            colors = {'Test confidence quantiles': 'darkblue',
                      'Uniform train confidence quantiles distribution': 'purple'}
            labels = list(colors.keys())
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
            plt.legend(handles, labels)
            plt.ylabel('Density')
            plt.xlabel('Confidence Quantile')
            plt.title('Distribution of samples confidence per confidence quantile')
            plt.show()

        footnote = '<p style="font-size:0.9em"><i>Explain here</i></p>'
        display = [display_plot, footnote, '<h5>Top Confidence Samples Indexes</h5>', top_k,
                   '<h5>Worst Confidence Samples Indexes</h5>', bottom_k]
        return CheckResult(tested_confidence_mean, check=self.__class__, display=display)

