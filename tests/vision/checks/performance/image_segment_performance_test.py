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
from tests.checks.utils import equal_condition_result
from deepchecks.vision.checks.performance import ImageSegmentPerformance

import numpy as np
from hamcrest import assert_that, has_length, has_entries, has_items, close_to
from tests.vision.vision_conftest import *


def test_mnist(mnist_dataset_train, mock_trained_mnist):
    # Act
    result = ImageSegmentPerformance().run(mnist_dataset_train, mock_trained_mnist)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(5),
        'Area': has_length(1),
        'Aspect Ratio': has_items(has_entries({
            'start': 1.0, 'stop': np.inf, 'count': 60000, 'display_range': '[1, inf)',
            'metrics': has_entries({'Precision': close_to(0.982, 0.001), 'Recall': close_to(0.979, 0.001)})
        })),
    }))


def test_coco_and_condition(coco_train_visiondata, mock_trained_yolov5_object_detection):
    # Arrange
    check = ImageSegmentPerformance().add_condition_score_from_mean_ratio_not_less_than(0.5)\
        .add_condition_score_from_mean_ratio_not_less_than(0.1)
    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection)
    # Assert result
    assert_that(result.value, has_entries({
        'Normalized Blue Mean': has_length(5),
        'Normalized Green Mean': has_length(5),
        'Normalized Red Mean': has_length(5),
        'Brightness': has_length(5),
        'Area': has_length(4),
        'Aspect Ratio': has_items(
            has_entries({
                'start': -np.inf, 'stop': 0.6671875, 'count': 12, 'display_range': '(-inf, 0.67)',
                'metrics': has_entries({'AP': close_to(0.454, 0.001), 'AR': close_to(0.45, 0.001)})
            }),
            has_entries({
                'start': 0.6671875, 'stop': 0.75, 'count': 11, 'display_range': '[0.67, 0.75)',
                'metrics': has_entries({'AP': close_to(0.364, 0.001), 'AR': close_to(0.366, 0.001)})
            }),
            has_entries({
                'start': 0.75, 'stop': close_to(1.102, 0.001), 'count': 28, 'display_range': '[0.75, 1.1)',
                'metrics': has_entries({'AP': close_to(0.299, 0.001), 'AR': close_to(0.333, 0.001)})
            }),
            has_entries({
                'start': close_to(1.102, 0.001), 'stop': np.inf, 'count': 13, 'display_range': '[1.1, inf)',
                'metrics': has_entries({'AP': close_to(0.5, 0.001), 'AR': close_to(0.549, 0.001)})
            }),
        )
    }))
    # Assert condition
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            name='No segment with ratio between score to mean less than 10%'
        ),
        equal_condition_result(
            is_pass=False,
            name='No segment with ratio between score to mean less than 50%',
            details="Properties with failed segments: Normalized Green Mean: "
                    "{'Range': '[0.34, 0.366)', 'Metric': 'AP', 'Ratio': 0.44}"
        )
    ))
