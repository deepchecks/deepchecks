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
from deepchecks.core.check_utils.single_feature_contribution_utils import get_single_feature_contribution
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.core.check_utils.whole_dataset_drift_utils import run_whole_dataset_drift
import pandas as pd

__all__ = ['SimpleFeatureContributionTrainTest']

# TODO
from deepchecks.vision.utils import image_formatters

pps_url = 'https://docs.deepchecks.com/en/stable/examples/tabular/' \
          'checks/methodology/single_feature_contribution_train_test' \
          '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'

DEFAULT_IMAGE_PROPERTIES = ['aspect_ratio',
                            'blur',
                            'rms_contrast',
                            'area',
                            'brightness',
                            'normalized_red_mean',
                            'normalized_green_mean',
                            'normalized_blue_mean']


class SimpleFeatureContributionTrainTest(TrainTestCheck):
    """
    Return the Predictive Power Score of all features, in order to estimate each feature's ability to predict the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability of each feature to predict the label.
    A high PPS (close to 1) can mean that this feature's success in predicting the label is actually due to data
    leakage - meaning that the feature holds information that is based on the label to begin with.

    When we compare train PPS to test PPS, A high difference can strongly indicate leakage,
    as a feature that was "powerful" in train but not in test can be explained by leakage in train that does
    not affect a new dataset.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore predictor function
    n_show_top : int , default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    """

    """
    Calculate drift between the entire train and test datasets (based on image properties) using a trained model.

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
    random_state : int , default: 42
        Random seed for the check.
    test_size : float , default: 0.3
        Fraction of the combined datasets to use for the evaluation of the domain classifier.
    """

    def __init__(
            self,
            alternative_image_properties: List[str] = None,
            n_top_properties: int = 3,
            ppscore_params: dict = None,


    ):
        super().__init__()

        if alternative_image_properties:
            self.image_properties = alternative_image_properties
        else:
            self.image_properties = DEFAULT_IMAGE_PROPERTIES

        self.n_top_properties = n_top_properties
        self.ppscore_params = ppscore_params or {}

        self._train_properties = OrderedDict([(k, []) for k in self.image_properties])
        self._test_properties = OrderedDict([(k, []) for k in self.image_properties])
        self._train_properties['target'] = []
        self._test_properties['target'] = []

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self._train_properties
        else:
            dataset = context.test
            properties = self._test_properties

        imgs = dataset.batch_to_images(batch)
        for func_name in self.image_properties:
            properties[func_name] += getattr(image_formatters, func_name)(imgs)

        properties['target'] += dataset.batch_to_labels(batch)

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

        text = [
            'The Predictive Power Score (PPS) is used to estimate the ability of a feature to predict the '
            f'label by itself. (Read more about {pps_html})'
            '',
            '<u>In the graph above</u>, we should suspect we have problems in our data if:',
            ''
            '1. <b>Train dataset PPS values are high</b>:',
            'Can indicate that this feature\'s success in predicting the label is actually due to data leakage, ',
            '   meaning that the feature holds information that is based on the label to begin with.',
            '2. <b>Large difference between train and test PPS</b> (train PPS is larger):',
            '   An even more powerful indication of data leakage, as a feature that was powerful in train but not in '
            'test ',
            '   can be explained by leakage in train that is not relevant to a new dataset.',
            '3. <b>Large difference between test and train PPS</b> (test PPS is larger):',
            '   An anomalous value, could indicate  drift in test dataset that caused a coincidental correlation to '
            'the target label.'
        ]

        ret_value, display = get_single_feature_contribution(df_train,
                                                             'target',
                                                             df_test,
                                                             'target',
                                                             self.ppscore_params,
                                                             self.n_top_properties)

        if display:
            display += text

        return CheckResult(value=ret_value, display=display, header='Single Feature Contribution Train-Test')
