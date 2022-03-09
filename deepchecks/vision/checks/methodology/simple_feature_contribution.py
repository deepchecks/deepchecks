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
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.core.check_utils.whole_dataset_drift_utils import run_whole_dataset_drift
import pandas as pd

__all__ = ['SimpleFeatureContributionTrainTest']

from deepchecks.vision.utils import image_formatters
from deepchecks.vision.vision_data import TaskType
import numpy as np

pps_url = 'https://docs.deepchecks.com/en/stable/examples/vision/' \
          'checks/methodology/simple_feature_contribution' \
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


def crop_img(img: np.array, x: int, y: int, w: int, h: int) -> np.array:
    return img[y:y + h, x:x + w]


class SimpleFeatureContributionTrainTest(TrainTestCheck):
    """
    Return the Predictive Power Score of image properties, in order to estimate their ability to predict the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability to predict the label by an image property (e.g.
    brightness, contrast etc.)
    A high PPS (close to 1) can mean that there's a bias in the dataset, as a single property can predict the label
    successfully, using simple classic ML algorithms - meaning that a deep learning algorithm may accidentally learn
    these properties instead of more accurate complex abstractions.
    For example, in a classification dataset of wolves and dogs photographs, if only wolves are photographed in the
    snow, the brightness of the image may be used to predict the label "wolf" easily. In this case, a model might not
    learn to discern wolf from dog by the animal's characteristics, but by using the background color.

    When we compare train PPS to test PPS, A high difference can strongly indicate bias in the datasets,
    as a property that was "powerful" in train but not in test can be explained by bias in train that does
    not affect a new dataset.

    For classification tasks, this check uses PPS to predict the class by image properties.
    For object detection tasks, this check uses PPS to predict the class of each bounding box, by the image properties
    of that specific bounding box.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore predictor function
    n_show_top : int , default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
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

        if dataset.task_type == TaskType.CLASSIFICATION:
            imgs = dataset.batch_to_images(batch)
            properties['target'] += dataset.batch_to_labels(batch)
        elif dataset.task_type == TaskType.OBJECT_DETECTION:
            labels = dataset.batch_to_labels(batch)
            orig_imgs = dataset.batch_to_images(batch)

            classes = []
            imgs = []
            for img, label in zip(orig_imgs, labels):
                classes += [int(x[0]) for x in label]

                bboxes = [np.array(x[1:]).astype(int) for x in label]
                imgs += [crop_img(img, *bbox) for bbox in bboxes]

            properties['target'] += classes
        else:
            raise DeepchecksValueError(
                f'Check {self.__class__.__name__} does not support task type {dataset.task_type}')

        for func_name in self.image_properties:
            properties[func_name] += getattr(image_formatters, func_name)(imgs)

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
            'The Predictive Power Score (PPS) is used to estimate the ability of an image property (such as brightness)'
            f'to predict the label by itself. (Read more about {pps_html})'
            '',
            '<u>In the graph above</u>, we should suspect we have problems in our data if:',
            ''
            '1. <b>Train dataset PPS values are high</b>:',
            '   A high PPS (close to 1) can mean that there\'s a bias in the dataset, as a single property can predict' 
            '   the label successfully, using simple classic ML algorithms',
            '2. <b>Large difference between train and test PPS</b> (train PPS is larger):',
            '   An even more powerful indication of dataset bias, as an image property that was powerful in train',
            '   but not in test can be explained by bias in train that is not relevant to a new dataset.',
            '3. <b>Large difference between test and train PPS</b> (test PPS is larger):',
            '   An anomalous value, could indicate drift in test dataset that caused a coincidental correlation to '
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

        return CheckResult(value=ret_value, display=display, header='Simple Feature Contribution Train-Test')
