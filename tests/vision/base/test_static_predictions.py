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
#
import typing as t

import numpy as np
from hamcrest import (assert_that, close_to, contains_exactly, equal_to, greater_than, has_entries, has_items,
                      has_length, instance_of)
from plotly.basedatatypes import BaseFigure

from deepchecks.vision.checks.model_evaluation.class_performance import ClassPerformance
from deepchecks.vision.checks.model_evaluation.image_segment_performance import ImageSegmentPerformance
from deepchecks.vision.checks.model_evaluation.train_test_prediction_drift import TrainTestPredictionDrift
from deepchecks.vision.suites.default_suites import full_suite
from deepchecks.vision.vision_data import VisionData
from tests.base.utils import equal_condition_result
from tests.common import assert_class_performance_display
from tests.conftest import get_expected_results_length, validate_suite_result


def _create_static_predictions(train: VisionData, test: VisionData, model, device):
    static_preds = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_pred = {}
            for i, batch in enumerate(vision_data):
                predictions = vision_data.infer_on_batch(batch, model, device)
                indexes = list(vision_data.data_loader.batch_sampler)[i]
                static_pred.update(dict(zip(indexes, predictions)))
        else:
            static_pred = None
        static_preds.append(static_pred)
    train_preds, tests_preds = static_preds
    return train_preds, tests_preds


# copied from class_performance_test
def test_class_performance_mnist_largest(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    train_preds, tests_preds = _create_static_predictions(mnist_dataset_train, mnist_dataset_test,
                                                          mock_trained_mnist, device)
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test,
                       train_predictions=train_preds, test_predictions=tests_preds,
                       device=device, n_samples=None)

    # Assert
    assert_that(set(result.value['Class']), equal_to(set(range(10))))
    assert_that(len(result.value), equal_to(40))
    assert_that(result.display, has_length(greater_than(0)))

    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))

    assert_that(figure.data, assert_class_performance_display(
        xaxis=[
            contains_exactly(equal_to('2'), equal_to('1')),
            contains_exactly(equal_to('1'), equal_to('2')),
            contains_exactly(equal_to('2'), equal_to('1')),
            contains_exactly(equal_to('2'), equal_to('1')),
        ],
        yaxis=[
            contains_exactly(
                close_to(0.984, 0.001),
                close_to(0.977, 0.001),
            ),
            contains_exactly(
                close_to(0.974, 0.001),
                close_to(0.973, 0.001),
            ),
            contains_exactly(
                close_to(0.988, 0.001),
                close_to(0.987, 0.001),
            ),
            contains_exactly(
                close_to(0.984, 0.001),
                close_to(0.980, 0.001),
            )
        ]
    ))


# copied from class_performance_test but added a sample before
def test_class_performance_mnist_largest_sampled_before(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    sampled_train = mnist_dataset_train.copy(shuffle=True, n_samples=1000, random_state=42)
    sampled_test = mnist_dataset_test.copy(shuffle=True, n_samples=1000, random_state=42)
    train_preds, tests_preds = _create_static_predictions(sampled_train, sampled_test,
                                                          mock_trained_mnist, device)
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(sampled_train, sampled_test,
                       train_predictions=train_preds, test_predictions=tests_preds,
                       device=device, n_samples=None)

    # Assert
    assert_that(set(result.value['Class']), equal_to(set(range(10))))
    assert_that(len(result.value), equal_to(40))
    assert_that(result.display, has_length(greater_than(0)))

    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))

    assert_that(figure.data, assert_class_performance_display(
        metrics=('Precision', 'Recall'),
        xaxis=[
            contains_exactly(equal_to('2'), equal_to('3')),
            contains_exactly(equal_to('2'), equal_to('3')),
            contains_exactly(equal_to('2'), equal_to('3')),
            contains_exactly(equal_to('2'), equal_to('3')),
        ],
        yaxis=[
            contains_exactly(
                close_to(1.0, 0.001),
                close_to(0.990, 0.001),
            ),
            contains_exactly(
                close_to(1.0, 0.001),
                close_to(0.980, 0.001),
            ),
            contains_exactly(
                close_to(0.976, 0.001),
                close_to(0.966, 0.001),
            ),
            contains_exactly(
                close_to(0.991, 0.001),
                close_to(0.983, 0.001),
            )
        ]
    ))


# copied from class_performance_test but sampled
def test_class_performance_mnist_largest_sampled(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    train_preds, tests_preds = _create_static_predictions(mnist_dataset_train, mnist_dataset_test,
                                                          mock_trained_mnist, device)
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test,
                       train_predictions=train_preds, test_predictions=tests_preds,
                       device=device)

    # Assert
    assert_that(set(result.value['Class']), equal_to(set(range(10))))
    assert_that(len(result.value), equal_to(40))
    assert_that(result.display, has_length(greater_than(0)))

    figure = t.cast(BaseFigure, result.display[0])
    assert_that(figure, instance_of(BaseFigure))

    assert_that(figure.data, assert_class_performance_display(
        xaxis=[
            contains_exactly(equal_to('1'), equal_to('2')),
            contains_exactly(equal_to('2'), equal_to('1')),
            contains_exactly(equal_to('2'), equal_to('1')),
            contains_exactly(equal_to('2'), equal_to('1')),
        ],
        yaxis=[
            contains_exactly(
                close_to(0.987, 0.001),
                close_to(0.987, 0.001),
            ),
            contains_exactly(
                close_to(0.981, 0.001),
                close_to(0.972, 0.001),
            ),
            contains_exactly(
                close_to(0.988, 0.001),
                close_to(0.987, 0.001),
            ),
            contains_exactly(
                close_to(0.984, 0.001),
                close_to(0.980, 0.001),
            )
        ]
    ))


# copied from image_segment_performance_test
def test_image_segment_performance_coco_and_condition(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    train_preds, _ = _create_static_predictions(coco_train_visiondata, None,
                                                mock_trained_yolov5_object_detection, device)
    check = ImageSegmentPerformance().add_condition_score_from_mean_ratio_greater_than(0.5) \
        .add_condition_score_from_mean_ratio_greater_than(0.1)
    # Act
    result = check.run(coco_train_visiondata, predictions=train_preds)
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
            name='Segment\'s ratio between score to mean is greater than 50%',
            details="Properties with failed segments: Mean Green Relative Intensity: "
                    "{'Range': '[0.34, 0.366)', 'Metric': 'Average Precision', 'Ratio': 0.44}"
        )
    ))


# copied from train_test_prediction_drift_test
def test_train_test_prediction_with_drift_object_detection_change_max_cat(coco_train_visiondata, coco_test_visiondata,
                                                                          mock_trained_yolov5_object_detection, device):
    # Arrange
    train_preds, test_preds = _create_static_predictions(coco_train_visiondata,
                                                         coco_test_visiondata,
                                                         mock_trained_yolov5_object_detection,
                                                         device)
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories_for_drift=100,
                                     min_category_size_ratio=0)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata,
                       train_predictions=train_preds, test_predictions=test_preds, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.48, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.085, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_suite(coco_train_visiondata, coco_test_visiondata,
               mock_trained_yolov5_object_detection, device):
    train_preds, test_preds = _create_static_predictions(coco_train_visiondata,
                                                         coco_test_visiondata,
                                                         mock_trained_yolov5_object_detection,
                                                         device)

    args = dict(train_dataset=coco_train_visiondata, test_dataset=coco_test_visiondata,
                train_predictions=train_preds, test_predictions=test_preds)
    suite = full_suite()
    result = suite.run(**args)
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)
