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
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_utils.whole_dataset_drift_utils import run_whole_dataset_drift
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.image_properties import default_image_properties, get_column_type, validate_properties

__all__ = ['ImageDatasetDrift']


class ImageDatasetDrift(TrainTestCheck):
    """Calculate drift between the entire train and test datasets (based on image properties) using a trained model.

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
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of 'continuous'/'discrete'
    n_top_properties : int , default: 3
        Amount of properties to show ordered by domain classifier feature importance. This limit is used together
        (AND) with min_feature_importance, so less than n_top_columns features can be displayed.
    min_feature_importance : float , default: 0.05
        Minimum feature importance to show in the check display. The features are the image properties that are given
        to the Domain Classifier as features to learn on.
        Feature importance sums to 1, so for example the default value of 0.05 means that all features with importance
        contributing less than 5% to the predictive power of the Domain Classifier won't be displayed. This limit is
        used together (AND) with n_top_columns, so features more important than min_feature_importance can be hidden.
    sample_size : int , default: 10_000
        Max number of rows to use from each dataset for the training and evaluation of the domain classifier.
    test_size : float , default: 0.3
        Fraction of the combined datasets to use for the evaluation of the domain classifier.
    min_meaningful_drift_score : float , default 0.05
        Minimum drift score for displaying drift in check. Under that score, check will display "nothing found".
    """

    def __init__(
            self,
            image_properties: List[Dict[str, Any]] = None,
            n_top_properties: int = 3,
            min_feature_importance: float = 0.05,
            sample_size: int = 10_000,
            test_size: float = 0.3,
            min_meaningful_drift_score: float = 0.05,
            **kwargs
    ):
        super().__init__(**kwargs)
        if image_properties:
            validate_properties(image_properties)
            self.image_properties = image_properties
        else:
            self.image_properties = default_image_properties

        self.n_top_properties = n_top_properties
        self.min_feature_importance = min_feature_importance
        self.sample_size = sample_size
        self.test_size = test_size
        self.min_meaningful_drift_score = min_meaningful_drift_score

        self._train_properties = defaultdict(list)
        self._test_properties = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            properties = self._train_properties
        else:
            properties = self._test_properties
        images = batch.images

        for single_property in self.image_properties:
            properties[single_property['name']].extend(single_property['method'](images))

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

        headnote = """
        <span>
        The shown features are the image properties (brightness, aspect ratio, etc.) that are most important for the
        domain classifier - the domain_classifier trained to distinguish between the train and test datasets.<br>
        </span>
        """

        numeric_features = []
        categorical_features = []
        for prop in self.image_properties:
            col_type = get_column_type(prop['output_type'])
            if col_type == 'numerical':
                numeric_features.append(prop['name'])
            else:
                categorical_features.append(prop['name'])

        values_dict, displays = run_whole_dataset_drift(
            train_dataframe=df_train, test_dataframe=df_test, numerical_features=numeric_features,
            cat_features=categorical_features, sample_size=sample_size, random_state=context.random_state,
            test_size=self.test_size, n_top_columns=self.n_top_properties,
            min_feature_importance=self.min_feature_importance, max_num_categories_for_display=None,
            show_categories_by=None, min_meaningful_drift_score=self.min_meaningful_drift_score
        )

        if displays:
            displays.insert(0, headnote)

        return CheckResult(value=values_dict, display=displays, header='Image Dataset Drift')
