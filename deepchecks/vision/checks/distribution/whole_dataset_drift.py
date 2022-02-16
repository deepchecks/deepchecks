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
from collections import OrderedDict
from typing import Any, List

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.core.check_utils.whole_dataset_drift_utils import run_whole_dataset_drift
import pandas as pd

__all__ = ['WholeDatasetDrift']

DEFAULT_IMAGE_PROPERTIES = ['aspect_ratio',
                            'area',
                            'brightness',
                            'normalized_red_mean',
                            'normalized_green_mean',
                            'normalized_blue_mean']


class WholeDatasetDrift(TrainTestCheck):
    """
    Calculate drift between the entire train and test datasets using a model trained to distinguish between them.

    Check fits a new model to distinguish between train and test datasets, called a Domain Classifier.
    The Domain Classifier is a tabular model, that cannot run on the images themselves. Therefore, the check calculates
    properties for each image (such as brightness, aspect ratio etc.) and uses them as input features to the Domain
    Classifier.
    Once the Domain Classifier is fitted the check calculates the feature importance for the domain classifier
    model. The result of the check is based on the AUC of the domain classifier model, and the check displays
    the change in distribution between train and test for the top features according to the
    calculated feature importance.

    Parameters
    ----------
    alternative_image_properties : List[str] , default: None
        List of alternative image properties names. Must be attributes of the ImageFormatter classes that are passed to
        train and test's VisionData class. If None, check uses DEFAULT_IMAGE_PROPERTIES.
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
    """

    def __init__(
            self,
            alternative_image_properties: List[str] = None,
            n_top_columns: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories: int = 10,
            sample_size: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3
    ):
        super().__init__()

        if alternative_image_properties:
            self.image_properties = alternative_image_properties
        else:
            self.image_properties = DEFAULT_IMAGE_PROPERTIES

        self.n_top_columns = n_top_columns
        self.min_feature_importance = min_feature_importance
        self.max_num_categories = max_num_categories
        self.sample_size = sample_size
        self.random_state = random_state
        self.test_size = test_size

        self._train_properties = OrderedDict([(k, []) for k in self.image_properties])
        self._test_properties = OrderedDict([(k, []) for k in self.image_properties])

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self._train_properties
        else:
            dataset = context.test
            properties = self._test_properties

        imgs = dataset.image_transformer(batch[0])
        for func_name in self.image_properties:
            image_property_function = dataset.image_transformer.__getattribute__(func_name)
            properties[func_name] += image_property_function(imgs)

    def compute(self, context: Context) -> CheckResult:
        """Train a Domain Classifier on image property data that was collected during update() calls.

        Returns
        -------
        CheckResult
            value: dictionary containing the domain classifier auc and a dict of column name to its feature
            importance as calculated for the domain classifier model.
            display: distribution graph for each column for the columns most explaining the dataset difference,
            comparing the train and test distributions.
        """
        df_train = pd.DataFrame(self._train_properties)
        df_test = pd.DataFrame(self._test_properties)

        sample_size = min(self.sample_size, df_train.shape[0], df_test.shape[0])

        return run_whole_dataset_drift(
            train_dataframe=df_train, test_dataframe=df_test, numerical_features=self.image_properties, cat_features=[],
            sample_size=sample_size, random_state=self.random_state, test_size=self.test_size,
            n_top_columns=self.n_top_columns, min_feature_importance=self.min_feature_importance,
            max_num_categories=self.max_num_categories
        )
