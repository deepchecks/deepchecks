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
import numpy as np
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_items, has_length, matches_regexp
from ignite.metrics import Precision, Recall

from deepchecks.vision.checks import ImageSegmentPerformance
from tests.base.utils import equal_condition_result
from tests.vision.vision_conftest import *


def test_mnist(mnist_dataset_train, mock_trained_mnist, device):
    # Act
    result = ImageSegmentPerformance().run(mnist_dataset_train, mock_trained_mnist, n_samples=None, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(5),
        'Area': has_length(1),
        'Aspect Ratio': has_items(has_entries({
            'start': 1.0, 'stop': np.inf, 'count': 60000, 'display_range': '[1, inf)',
            'metrics': has_entries({'Precision': close_to(0.982, 0.001), 'Recall': close_to(0.979, 0.001)})
        })),
    }))
    assert_that(result.display, has_length(greater_than(0)))


def test_mnist_without_display(mnist_dataset_train, mock_trained_mnist, device):
    # Act
    result = ImageSegmentPerformance().run(mnist_dataset_train, mock_trained_mnist,
                                           n_samples=None, device=device, with_display=False)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(5),
        'Area': has_length(1),
        'Aspect Ratio': has_items(has_entries({
            'start': 1.0, 'stop': np.inf, 'count': 60000, 'display_range': '[1, inf)',
            'metrics': has_entries({'Precision': close_to(0.982, 0.001), 'Recall': close_to(0.979, 0.001)})
        })),
    }))
    assert_that(result.display, has_length(0))


def test_mnist_no_display(mnist_dataset_train, mock_trained_mnist, device):
    # Act
    result = ImageSegmentPerformance().run(mnist_dataset_train, mock_trained_mnist, n_samples=1, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(1),
        'Area': has_length(1),
        'Aspect Ratio': has_length(1),
    }))
    assert_that(result.display, has_length(1))
    assert_that(result.display[0], matches_regexp('<i>Note'))  #


def test_mnist_top_display(mnist_dataset_train, mock_trained_mnist, device):
    # Act
    result = ImageSegmentPerformance(n_to_show=1).run(mnist_dataset_train, mock_trained_mnist, n_samples=None,
                                                      device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(5),
        'Area': has_length(1),
        'Aspect Ratio': has_items(has_entries({
            'start': 1.0, 'stop': np.inf, 'count': 60000, 'display_range': '[1, inf)',
            'metrics': has_entries({'Precision': close_to(0.982, 0.001), 'Recall': close_to(0.979, 0.001)})
        })),
    }))
    assert_that(result.display, has_length(1))
    assert_that(result.display[0].data, has_length(1))
    assert_that(result.display[0].data[0].name, equal_to('Precision'))


def test_mnist_alt_metrics(mnist_dataset_train, mock_trained_mnist, device):
    # Act
    result = ImageSegmentPerformance(scorers={'p': Precision(), 'r': Recall()}) \
        .run(mnist_dataset_train, mock_trained_mnist, n_samples=None, device=device)
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_length(5),
        'Area': has_length(1),
        'Aspect Ratio': has_items(has_entries({
            'start': 1.0, 'stop': np.inf, 'count': 60000, 'display_range': '[1, inf)',
            'metrics': has_entries({'p': close_to(0.982, 0.001), 'r': close_to(0.979, 0.001)})
        })),
    }))
    assert_that(result.display, has_length(1))
    assert_that(result.display[0].data, has_length(4))
    assert_that(result.display[0].data[0].name, equal_to('p'))
    assert_that(result.display[0].data[3].name, equal_to('r'))


def test_coco_and_condition(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = ImageSegmentPerformance().add_condition_score_from_mean_ratio_greater_than(0.6) \
        .add_condition_score_from_mean_ratio_greater_than(0.1)
    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    # Assert result
    assert_that(result.value, has_entries({
        'Mean Blue Relative Intensity': has_length(5),
        'Mean Green Relative Intensity': has_length(5),
        'Mean Red Relative Intensity': has_length(5),
        'Brightness': has_length(5),
        'Area': has_length(4),
        'Aspect Ratio': has_items(
            has_entries({
                'start': -np.inf, 'stop': 0.6671875, 'count': 12, 'display_range': '(-inf, 0.67)',
                'metrics': has_entries({'Average Precision': close_to(0.454, 0.001), 'Average Recall': close_to(0.45, 0.001)})
            }),
            has_entries({
                'start': 0.6671875, 'stop': 0.75, 'count': 11, 'display_range': '[0.67, 0.75)',
                'metrics': has_entries({'Average Precision': close_to(0.367, 0.001), 'Average Recall': close_to(0.4, 0.001)})
            }),
            has_entries({
                'start': 0.75, 'stop': close_to(1.102, 0.001), 'count': 28, 'display_range': '[0.75, 1.1)',
                'metrics': has_entries({'Average Precision': close_to(0.299, 0.001), 'Average Recall': close_to(0.333, 0.001)})
            }),
            has_entries({
                'start': close_to(1.102, 0.001), 'stop': np.inf, 'count': 13, 'display_range': '[1.1, inf)',
                'metrics': has_entries({'Average Precision': close_to(0.5, 0.001), 'Average Recall': close_to(0.549, 0.001)})
            }),
        )
    }))
    # Assert condition
    assert_that(result.conditions_results, has_items(
        equal_condition_result(
            is_pass=True,
            name='Segment\'s ratio between score to mean is greater than 10%',
            details="Found minimum ratio for property Mean Green Relative Intensity: "
                    "{'Range': '[0.34, 0.366)', 'Metric': 'Average Precision', 'Ratio': 0.44}"
        ),
        equal_condition_result(
            is_pass=False,
            name='Segment\'s ratio between score to mean is greater than 60%',
            details="Properties with failed segments: "
                    "Brightness: {'Range': '[0.48, 0.54)', 'Metric': 'Average Recall', 'Ratio': 0.55}, "
                    "Mean Green Relative Intensity: {'Range': '[0.34, 0.366)', 'Metric': 'Average Precision', "
                    "'Ratio': 0.44}"
        )
    ))


def test_segmentation_coco_and_condition(segmentation_coco_train_visiondata,
                                         trained_segmentation_deeplabv3_mobilenet_model, device):
    # Arrange
    check = ImageSegmentPerformance()
    # Act
    result = check.run(segmentation_coco_train_visiondata, trained_segmentation_deeplabv3_mobilenet_model, device=device)
    # Assert result
    assert_that(result.value, has_entries({
        'Mean Blue Relative Intensity': has_length(5),
        'Mean Green Relative Intensity': has_length(5),
        'Mean Red Relative Intensity': has_length(5),
        'Brightness': has_length(5),
        'Area': has_length(5),
        'Aspect Ratio': has_items(
            has_entries({
                'start': -np.inf, 'stop': close_to(0.73, 0.01), 'count': 2, 'display_range': '(-inf, 0.73)',
                'metrics': has_entries({'Dice': close_to(0.9, 0.01)})
            }),
            has_entries({
                'start': close_to(0.73, 0.01), 'stop': close_to(0.76, 0.01), 'count': 2, 'display_range': '[0.73, 0.76)',
                'metrics': has_entries({'Dice': close_to(0.94, 0.01)})
            })
        )
    }))
