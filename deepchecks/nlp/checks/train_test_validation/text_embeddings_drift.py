# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
"""Module contains the embeddings drift check."""

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.nlp import Context, TrainTestCheck
from deepchecks.nlp.utils.multivariate_embeddings_drift_utils import run_multivariable_drift_for_embeddings
from deepchecks.utils.strings import format_number

__all__ = ['TextEmbeddingsDrift']


class TextEmbeddingsDrift(TrainTestCheck):
    """
    Calculate drift between the train and test datasets using a model trained to distinguish between their embeddings.

    This check detects drift between the model embeddings of the train and test data. To do so, the check trains
    a Domain Classifier, which is a model trained to distinguish between the train and test datasets.

    For optimizing time and improving the model's performance, the check uses dimension reduction to reduce the
    number of embeddings dimensions. The check uses UMAP for dimension reduction by default, but can also use PCA
    or no dimension reduction at all.

    For more information about embeddings in deepchecks, see :ref:`Text Embeddings Guide <nlp__embeddings_guide>`.

    Parameters
    ----------
    sample_size : int , default: 10_000
        Max number of rows to use from each dataset for the training and evaluation of the domain classifier.
    random_state : int , default: 42
        Random seed for the check.
    test_size : float , default: 0.3
        Fraction of the combined datasets to use for the evaluation of the domain classifier
    dimension_reduction_method : str , default: 'auto'
        Dimension reduction method to use for the check. Dimension reduction is used to reduce the number of
        embeddings dimensions in order for the domain classifier to train more efficiently on the data.
        The 2 supported methods are PCA and UMAP. While UMAP yields better results (especially visually), it is much
        slower than PCA.
        Supported values:
        - 'auto' (default): Automatically choose the best method for the data. Uses UMAP if with_display is True,
        otherwise uses PCA for a faster calculation. Doesn't use dimension reduction at all if the number of embeddings
        dimensions is less than 30.
        - 'pca': Use PCA for dimension reduction.
        - 'umap': Use UMAP for dimension reduction.
        - 'none': Don't use dimension reduction.
    num_samples_in_display : int , default: 500
        Number of samples to display in the check display scatter plot.
    """

    def __init__(
            self,
            sample_size: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3,
            dimension_reduction_method: str = 'auto',
            num_samples_in_display: int = 500,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.sample_size = sample_size
        self.random_state = random_state
        self.test_size = test_size
        if dimension_reduction_method is None:
            dimension_reduction_method = 'none'
        if dimension_reduction_method.lower() not in ['auto', 'umap', 'pca', 'none']:
            raise ValueError(f'dimension_reduction_method must be one of "auto", "umap", "pca" or "none". '
                             f'Got {dimension_reduction_method} instead')
        self.dimension_reduction_method = dimension_reduction_method.lower()
        self.num_samples_in_display = num_samples_in_display

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value: dictionary containing the domain classifier auc and a dict of column name to its feature
            importance as calculated for the domain classifier model.
            display: distribution graph for each column for the columns most explaining the dataset difference,
            comparing the train and test distributions.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset or DataFrame instance
        """
        train_dataset = context.train
        test_dataset = context.test

        sample_size = min(self.sample_size, train_dataset.n_samples, test_dataset.n_samples)

        values_dict, displays = run_multivariable_drift_for_embeddings(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            sample_size=sample_size, random_state=self.random_state,
            test_size=self.test_size,
            num_samples_in_display=self.num_samples_in_display,
            dimension_reduction_method=self.dimension_reduction_method,
            with_display=context.with_display,
            model_classes=context.model_classes
        )

        return CheckResult(value=values_dict, display=displays, header='Embeddings Drift')

    def add_condition_overall_drift_value_less_than(self, max_drift_value: float = 0.25):
        """Add condition.

        Overall drift score, calculated as (2 * AUC - 1) for the AUC of the dataset discriminator model, is less
        than the specified value. This value is used as it scales the AUC value to the range [0, 1], where 0 indicates
        a random model (and no drift) and 1 indicates a perfect model (and completely distinguishable datasets).

        Parameters
        ----------
        max_drift_value : float , default: 0.25
            Maximal drift value allowed (value 0 and above)
        """

        def condition(result: dict):
            drift_score = result['domain_classifier_drift_score']
            details = f'Found drift value of: {format_number(drift_score)}, corresponding to a domain classifier ' \
                      f'AUC of: {format_number(result["domain_classifier_auc"])}'
            category = ConditionCategory.PASS if drift_score < max_drift_value else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Drift value is less than {format_number(max_drift_value)}',
                                  condition)
