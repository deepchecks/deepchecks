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
import plotly.graph_objects as go

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult, ConditionCategory
from deepchecks.utils.distribution.trust_score import TrustScore
from deepchecks.utils.distribution.preprocessing import ScaledNumerics
from deepchecks.utils.distribution.plot import feature_distribution_traces
from deepchecks.utils.metrics import task_type_check, ModelType
from deepchecks.utils.strings import format_percent
from deepchecks.utils.validation import validate_model
from deepchecks.errors import DeepchecksValueError, ModelValidationError, DatasetValidationError


__all__ = ['TrustScoreComparison']


class TrustScoreComparison(TrainTestBaseCheck):
    """Compares the model's trust score for the train dataset with scores of the test dataset.

    The Trust Score algorithm and code was published in the paper: "To Trust or not to trust c classifier". See the
    original paper at arxiv 1805.11783, or see the version of the paper presented at NeurIPS in 2018:
    https://proceedings.neurips.cc/paper/2018/file/7180cffd6a8e829dacfc2a31b3f72ece-Paper.pdf

    The process is as follows:

    #. Pre-process the train and test data into scaled numerics.
    #. Train a TrustScore regressor based on train data + label.
    #. Predict on test data using the model.
    #. Use TrustScore to score the prediction of the model.

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
                 random_state: int = 42, n_to_show: int = 5, percent_top_scores_to_hide: float = 0.05):
        super().__init__()
        _validate_parameters(k_filter, alpha, max_number_categories, min_test_samples, sample_size, n_to_show,
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
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset, cast=True)
        test_label = self._dataset_has_label(test_dataset)
        validate_model(test_dataset, model)
        model_type = task_type_check(model, test_dataset)

        # Baseline must have label so we must get it as Dataset.
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset)
        train_label = self._dataset_has_label(train_dataset)

        features_list = self._datasets_share_features([train_dataset, test_dataset])
        label_name = self._datasets_share_label([train_dataset, test_dataset])

        if test_dataset.n_samples < self.min_test_samples:
            raise DatasetValidationError(
                'Number of samples in test dataset has not passed the minimum. '
                'You can change the minimum number of samples required for the '
                'check to run with the parameter "min_test_samples"'
            )

        if model_type not in {ModelType.BINARY, ModelType.MULTICLASS}:
            raise ModelValidationError(
                'Check is relevant only for the classification models, but'
                f'received model of type {model_type.value.lower()}'
            )

        no_null_label_train = train_dataset.data[train_label.notna()]
        train_data_sample = no_null_label_train.sample(
            min(self.sample_size, len(no_null_label_train)),
            random_state=self.random_state
        )

        no_null_label_test = test_dataset.data[test_label.notna()]
        test_data_sample = no_null_label_test.sample(
            min(self.sample_size, len(no_null_label_test)),
            random_state=self.random_state
        )

        sn = ScaledNumerics(test_dataset.cat_features, self.max_number_categories)
        x_train = sn.fit_transform(train_data_sample[features_list])
        x_test = sn.transform(test_data_sample[features_list])

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
        y_train_pred = model.predict(train_data_sample[features_list]).flatten()
        train_trust_scores = trust_score_model.score(x_train.to_numpy(),
                                                     transform_numpy_label(y_train_pred))[0].astype('float64')
        # Calculate y on test dataset using the model
        y_test_pred = model.predict(test_data_sample[features_list]).flatten()
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

        headnote = r"""<span> Trust score roughly measures the following quantity:<br><br> <p> $$Trust Score = \frac{
        \textrm{Distance from the sample to the nearest training samples belonging to a class different than the
        predicted class}}{\textrm{Distance from the sample to the nearest training samples belonging to the predicted
        class}}$$ </p> So that higher values represent samples that are "close" to training examples with the same
        label as sample prediction, and lower values represent samples that are "far" from training samples with
        labels matching their prediction. For more information, please refer to the original paper at <a
        href="https://arxiv.org/abs/1805.11783"  target="_blank">arxiv 1805.11783</a>, or see the version of the <a
        href="https://proceedings.neurips.cc/paper/2018/file/7180cffd6a8e829dacfc2a31b3f72ece-Paper.pdf"
        target="_blank">paper presented at NeurIPS in 2018</a>.</span>"""

        footnote = """<span style="font-size:0.8em"><i>
            The test trust score distribution should be quite similar to the train's. If it is skewed to the left, the
            confidence of the model in the test data is lower than the train, indicating a difference that may affect
            model performance on similar data. If it is skewed to the right, it indicates an underlying problem with the
            creation of the test dataset (test confidence isn't expected to be higher than train's).
            </i></span>"""
        display = [headnote, _display_plot(train_trust_scores, test_trust_scores, self.percent_top_scores_to_hide),
                   footnote,
                   '<h5>Worst Trust Score Samples</h5>', bottom_k,
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
                return ConditionResult(False, message, category=ConditionCategory.WARN)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Mean trust score decline is not greater than {format_percent(threshold)}',
                                  condition)


def _is_positive_int(x) -> bool:
    return x is not None and isinstance(x, int) and x > 0


def _is_float_0_to_1(x) -> bool:
    return x is not None and isinstance(x, float) and 0 <= x <= 1


def _validate_parameters(k_filter, alpha, max_number_categories, min_test_samples, sample_size, n_to_show,
                         percent_top_scores_to_hide):
    if not _is_positive_int(k_filter):
        raise DeepchecksValueError(f'k_filter must be positive integer but got: {k_filter}')
    if not _is_float_0_to_1(alpha):
        raise DeepchecksValueError(f'alpha must be float between 0 to 1 but got: {alpha}')
    if not _is_positive_int(max_number_categories):
        raise DeepchecksValueError(f'max_number_categories must be positive integer but got: {max_number_categories}')
    if not _is_positive_int(min_test_samples):
        raise DeepchecksValueError(f'min_test_samples must be positive integer but got: {min_test_samples}')
    if not _is_positive_int(sample_size):
        raise DeepchecksValueError(f'sample_size must be positive integer but got: {min_test_samples}')
    if sample_size < min_test_samples:
        raise DeepchecksValueError('sample_size can\'t be smaller than min_test_samples')
    if not _is_positive_int(n_to_show):
        raise DeepchecksValueError(f'n_to_show must be positive integer but got: {min_test_samples}')
    if not _is_float_0_to_1(percent_top_scores_to_hide):
        raise DeepchecksValueError(f'percent_top_scores_to_hide must be float between 0 to 1 but got: '
                                   f'{percent_top_scores_to_hide}')


def _display_plot(train_trust_scores, test_trust_scores, percent_to_cut):
    """Display a distribution comparison plot for the given columns."""
    traces, xaxis, yaxis = feature_distribution_traces(train_trust_scores, test_trust_scores,
                                                       quantile_cut=percent_to_cut)
    xaxis['title'] = 'Trust Score'

    figure = go.Figure(layout=go.Layout(
        title='Trust Score Distribution',
        xaxis=xaxis,
        yaxis=yaxis,
        legend=dict(title='Dataset'),
        width=700,
        height=400
    ))

    figure.add_traces(traces)

    return figure
