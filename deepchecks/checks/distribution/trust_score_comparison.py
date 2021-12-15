# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module of trust score comparison check."""
import numpy as np
import matplotlib.pyplot as plt

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.checks.distribution.trust_score import TrustScore
from deepchecks.checks.distribution.preprocessing import preprocess_dataset_to_scaled_numerics
from deepchecks.checks.distribution.plot import plot_density
from deepchecks.utils.metrics import task_type_check, ModelType
from deepchecks.utils.strings import format_percent
from deepchecks.utils.validation import validate_model
from deepchecks.utils.plot import colors
from deepchecks.errors import DeepchecksValueError


__all__ = ['TrustScoreComparison']


def is_positive_int(x):
    return x is not None and isinstance(x, int) and x > 0


def is_float_0_to_1(x):
    return x is not None and isinstance(x, float) and 0 <= x <= 1


def validate_parameters(k_filter, alpha, max_number_categories, min_test_samples, sample_size, n_to_show,
                        percent_top_scores_to_hide):
    if not is_positive_int(k_filter):
        raise DeepchecksValueError(f'k_filter must be positive integer but got: {k_filter}')
    if not is_float_0_to_1(alpha):
        raise DeepchecksValueError(f'alpha must be float between 0 to 1 but got: {alpha}')
    if not is_positive_int(max_number_categories):
        raise DeepchecksValueError(f'max_number_categories must be positive integer but got: {max_number_categories}')
    if not is_positive_int(min_test_samples):
        raise DeepchecksValueError(f'min_test_samples must be positive integer but got: {min_test_samples}')
    if not is_positive_int(sample_size):
        raise DeepchecksValueError(f'sample_size must be positive integer but got: {min_test_samples}')
    if sample_size < min_test_samples:
        raise DeepchecksValueError('sample_size can\'t be smaller than min_test_samples')
    if not is_positive_int(n_to_show):
        raise DeepchecksValueError(f'n_to_show must be positive integer but got: {min_test_samples}')
    if not is_float_0_to_1(percent_top_scores_to_hide):
        raise DeepchecksValueError(f'percent_top_scores_to_hide must be float between 0 to 1 but got: '
                                   f'{percent_top_scores_to_hide}')


class TrustScoreComparison(TrainTestBaseCheck):
    """Compares the model's trust scores of the train dataset with scores of the test dataset.

    The process is as follows:
    * Pre-process the train and test data into scaled numerics.
    * Train a TrustScore (https://arxiv.org/abs/1805.11783) regressor based on train data + label.
    * Predict on test data using the model.
    * Use TrustScore to score the prediction of the model.

    Args:
        k_filter (int): used in TrustScore (Number of neighbors used during either kNN distance or probability
                        filtering)
        alpha (float): used in TrustScore (Fraction of instances to filter out to reduce impact of outliers)
        max_number_categories (int): Indicates the maximum number of unique categories in a single categorical
                                     column (rare categories will be changed to a form of "other")
        min_test_samples (int): Minimal number of samples in train data to be able to run this check
        sample_size (int): Number of samples to use for the check for train and test. if dataset contains less than
                           sample_size than all the dataset will be used.
        random_state (int): The random state to use for sampling.
        n_to_show (int): Number of samples to show of worst and best trust score.
    """

    def __init__(self, k_filter: int = 10, alpha: float = 0.001,
                 max_number_categories: int = 10, min_test_samples: int = 300, sample_size: int = 10_000,
                 random_state: int = 42, n_to_show: int = 5, percent_top_scores_to_hide: float = 0.01):
        super().__init__()
        validate_parameters(k_filter, alpha, max_number_categories, min_test_samples, sample_size, n_to_show,
                            percent_top_scores_to_hide)
        self.k_filter = k_filter
        self.alpha = alpha
        self.max_number_categories = max_number_categories
        self.min_test_samples = min_test_samples
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_to_show = n_to_show
        self.percent_top_scores_to_hide = percent_top_scores_to_hide

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): Dataset to use for TrustScore regressor
            test_dataset (Dataset): Dataset to check for trust score
            model: Model used to predict on the validation dataset
        """
        # tested dataset can be also dataframe
        test_dataset: Dataset = Dataset.validate_dataset_or_dataframe(test_dataset)
        validate_model(test_dataset, model)
        model_type = task_type_check(model, test_dataset)

        # Baseline must have label so we must get it as Dataset.
        Dataset.validate_dataset(train_dataset)
        train_dataset.validate_label()
        train_dataset.validate_shared_features(test_dataset)

        if test_dataset.n_samples < self.min_test_samples:
            msg = ('Number of samples in test dataset have not passed the minimum. you can change '
                   'minimum samples needed to run with parameter "min_test_samples"')
            raise DeepchecksValueError(msg)
        if model_type not in [ModelType.BINARY, ModelType.MULTICLASS]:
            raise DeepchecksValueError('Check supports only classification')

        train_data_sample = train_dataset.data.sample(min(self.sample_size, train_dataset.n_samples),
                                                      random_state=self.random_state)
        test_data_sample = test_dataset.data.sample(min(self.sample_size, test_dataset.n_samples),
                                                    random_state=self.random_state)
        features_list = train_dataset.features
        label_name = train_dataset.label_name

        x_train, x_test = preprocess_dataset_to_scaled_numerics(
            baseline_features=train_data_sample[features_list],
            test_features=test_data_sample[features_list],
            categorical_columns=test_dataset.cat_features,
            max_num_categories=self.max_number_categories
        )

        # Trust Score model expects labels to be consecutive integers from 0 to n-1, so we transform our label to
        # this format.
        label_transform_dict = dict(zip(sorted(list(set(train_data_sample[label_name]))),
                                        range(len(set(train_data_sample[label_name])))))

        def transform_numpy_label(np_label: np.ndarray) -> np.ndarray:
            return np.array([label_transform_dict[lbl] for lbl in np_label])

        y_train = train_data_sample[label_name].replace(label_transform_dict)
        trust_score_model = TrustScore(k_filter=self.k_filter, alpha=self.alpha)
        trust_score_model.fit(X=x_train.to_numpy(), Y=y_train.to_numpy())
        # Calculate y on train and get scores
        y_train_pred = model.predict(train_data_sample[features_list])
        train_trust_scores = trust_score_model.score(x_train.to_numpy(),
                                                     transform_numpy_label(y_train_pred))[0].astype('float64')
        # Calculate y on test dataset using the model
        y_test_pred = model.predict(test_data_sample[features_list])
        test_trust_scores = trust_score_model.score(x_test.to_numpy(),
                                                    transform_numpy_label(y_test_pred))[0].astype('float64')

        # Move label and index to the beginning if exists
        if test_dataset.label_name:
            label = test_dataset.label_name
            test_data_sample.insert(0, label, test_data_sample.pop(label))
        if test_dataset.index_name:
            index = test_dataset.index_name
            test_data_sample.insert(0, index, test_data_sample.pop(index))
        # Add score and prediction and sort by score
        test_data_sample.insert(0, 'Model Prediction', y_test_pred)
        test_data_sample.insert(0, 'Trust Score', test_trust_scores)
        test_data_sample = test_data_sample.sort_values(by=['Trust Score'], ascending=False)

        # Display top and bottom
        top_k = test_data_sample.head(self.n_to_show)
        bottom_k = test_data_sample.tail(self.n_to_show)

        def display_plot(percent_to_cut=self.percent_top_scores_to_hide):
            _, axes = plt.subplots(1, 1, figsize=(7, 4))

            def filter_quantile(data):
                return data[data < np.quantile(data, 1 - percent_to_cut)]

            test_trust_scores_cut = filter_quantile(test_trust_scores)
            train_trust_scores_cut = filter_quantile(train_trust_scores)
            x_range = [min(*test_trust_scores_cut, *train_trust_scores_cut),
                     max(*test_trust_scores_cut, *train_trust_scores_cut)]
            xs = np.linspace(x_range[0], x_range[1], 40)
            plot_density(train_trust_scores_cut, xs, colors['Train'])
            plot_density(test_trust_scores_cut, xs, colors['Test'])
            # Set x axis
            axes.set_xlim(x_range)
            plt.xlabel('Trust score')
            # Set y axis
            axes.set_ylim(bottom=0)
            plt.ylabel('Probability Density')
            # Set labels

            labels = list(colors.keys())
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
            plt.legend(handles, labels)
            plt.title('Trust Score Distribution')

        headnote = """<span>
        Trust score measures the agreement between the classifier and a modified nearest-neighbor
        classifier on the testing example. Higher values represent samples that are "close" to training examples with
        the same label as sample prediction, and lower values represent samples that are "far" from training samples
        with labels matching their prediction. (arxiv 1805.11783)
        </span>"""
        footnote = """<span style="font-size:0.8em"><i>
            The test trust score distribution should be quite similar to the train's. If it is skewed to the left, the
            confidence of the model in the test data is lower than the train, indicating a difference that may affect
            model performance on similar data. If it is skewed to the right, it indicates an underlying problem with the creation of the test dataset
            (test confidence isn't expected to be higher than train's).
            </i></span>"""
        display = [headnote, display_plot, footnote, '<h5>Worst Trust Score Samples</h5>', bottom_k,
                   '<h5>Top Trust Score Samples</h5>', top_k]

        result = {'test': np.mean(test_trust_scores), 'train': np.mean(train_trust_scores)}
        return CheckResult(result, display=display, header='Trust Score Comparison: Train vs. Test')

    def add_condition_mean_score_percent_decline_not_greater_than(self, threshold: float = 0.2):
        """Add condition.

        Percent of decline between the mean trust score of train and the mean trust score of test is not above
        given threshold.

        Args:
            threshold (float): Maximum percentage decline allowed (value 0 and above)
        """
        def condition(result: dict):
            train_score = result['train']
            test_score = result['test']
            pct_diff = (train_score - test_score) / train_score

            if pct_diff > threshold:
                message = f'Found decline of: {format_percent(-pct_diff)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Mean trust score decline is not greater than {format_percent(threshold)}',
                                  condition)
