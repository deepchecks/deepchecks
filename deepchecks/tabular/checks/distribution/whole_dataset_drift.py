# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains the domain classifier drift check."""
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.core.check_utils.whole_dataset_drift_utils import run_whole_dataset_drift
from deepchecks.utils.strings import format_number

__all__ = ['WholeDatasetDrift']


class WholeDatasetDrift(TrainTestCheck):
    """
    Calculate drift between the entire train and test datasets using a model trained to distinguish between them.

    Check fits a new model to distinguish between train and test datasets, called a Domain Classifier.
    Once the Domain Classifier is fitted the check calculates the feature importance for the domain classifier
    model. The result of the check is based on the AUC of the domain classifier model, and the check displays
    the change in distribution between train and test for the top features according to the
    calculated feature importance.

    Parameters
    ----------
    n_top_columns : int , default: 3
        Amount of columns to show ordered by domain classifier feature importance. This limit is used together
        (AND) with min_feature_importance, so less than n_top_columns features can be displayed.
    min_feature_importance : float , default: 0.05
        Minimum feature importance to show in the check display. Feature importance
        sums to 1, so for example the default value of 0.05 means that all features with importance contributing
        less than 5% to the predictive power of the Domain Classifier won't be displayed. This limit is used
        together (AND) with n_top_columns, so features more important than min_feature_importance can be
        hidden.
    max_num_categories : int , default: 10
        Only for categorical columns. Max number of categories to display in distributio plots. If there are
        more, they are binned into an "Other" category in the display. If max_num_categories=None, there is
        no limit.
    sample_size : int , default: 10_000
        Max number of rows to use from each dataset for the training and evaluation of the domain classifier.
    random_state : int , default: 42
        Random seed for the check.
    test_size : float , default: 0.3
        Fraction of the combined datasets to use for the evaluation of the domain classifier.
    min_meaningful_drift_score : float , default 0.05
        Minimum drift score for displaying drift in check. Under that score, check will display "nothing found".
    """

    def __init__(
            self,
            n_top_columns: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories: int = 10,
            sample_size: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3,
            min_meaningful_drift_score: float = 0.05
    ):
        super().__init__()

        self.n_top_columns = n_top_columns
        self.min_feature_importance = min_feature_importance
        self.max_num_categories = max_num_categories
        self.sample_size = sample_size
        self.random_state = random_state
        self.test_size = test_size
        self.min_meaningful_drift_score = min_meaningful_drift_score

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
        features = train_dataset.features
        cat_features = train_dataset.cat_features

        sample_size = min(self.sample_size, train_dataset.n_samples, test_dataset.n_samples)

        numerical_features = list(set(features) - set(cat_features))

        headnote = """
        <span>
        The shown features are the features that are most important for the domain classifier - the
        domain_classifier trained to distinguish between the train and test datasets.<br>
        </span>
        """

        values_dict, displays = run_whole_dataset_drift(train_dataframe=train_dataset.data[features],
                                                        test_dataframe=test_dataset.data[features],
                                                        numerical_features=numerical_features,
                                                        cat_features=cat_features,
                                                        sample_size=sample_size, random_state=self.random_state,
                                                        test_size=self.test_size, n_top_columns=self.n_top_columns,
                                                        min_feature_importance=self.min_feature_importance,
                                                        max_num_categories=self.max_num_categories,
                                                        min_meaningful_drift_score=self.min_meaningful_drift_score)

        if displays:
            displays.insert(0, headnote)

        return CheckResult(value=values_dict, display=displays, header='Whole Dataset Drift')

    def add_condition_overall_drift_value_not_greater_than(self, max_drift_value: float = 0.25):
        """Add condition.

        Overall drift score, calculated as (2 * AUC - 1) for the AUC of the dataset discriminator model, is not greater
        than the specified value. This value is used as it scales the AUC value to the range [0, 1], where 0 indicates
        a random model (and no drift) and 1 indicates a perfect model (and completely distinguishable datasets).

        Parameters
        ----------
        max_drift_value : float , default: 0.25
            Maximal drift value allowed (value 0 and above)
        """

        def condition(result: dict):
            drift_score = result['domain_classifier_drift_score']
            if drift_score > max_drift_value:
                message = f'Found drift value of: {format_number(drift_score)}, corresponding to a domain classifier ' \
                          f'AUC of: {format_number(result["domain_classifier_auc"])}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Drift value is not greater than {format_number(max_drift_value)}',
                                  condition)
