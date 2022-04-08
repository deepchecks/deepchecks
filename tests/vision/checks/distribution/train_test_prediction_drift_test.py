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
"""Test functions of the VISION train test prediction drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling
from tests.checks.utils import equal_condition_result
import torch.nn as nn
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import TrainTestPredictionDrift


def test_no_drift_classification(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(train, test, mock_trained_mnist,
                       device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        )}
    ))


def test_no_drift_object_detection(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_train_visiondata, mock_trained_yolov5_object_detection,
                       device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_classification(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(train, test, mock_trained_mnist,
                       device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0, 0.001),
             'Method': equal_to('PSI')}
        )
        }
    ))


def test_with_drift_object_detection(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection,
                                     device):
    # Arrange
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.31, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.009, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.054, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_change_max_cat(coco_train_visiondata, coco_test_visiondata,
                                                    mock_trained_yolov5_object_detection, device):
    # Arrange
    check = TrainTestPredictionDrift(max_num_categories=100)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.48, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.009, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.054, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_alternative_measurements(coco_train_visiondata, coco_test_visiondata,
                                                              mock_trained_yolov5_object_detection, device):
    # Arrange
    def prop(predictions):
        return [int(x[0][0]) if len(x) != 0 else 0 for x in predictions]
    alternative_measurements = [
        {'name': 'test', 'method': prop, 'output_type': 'continuous'}]
    check = TrainTestPredictionDrift(prediction_properties=alternative_measurements)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.037, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_drift_max_drift_score_condition_fail(mnist_drifted_datasets, mock_trained_mnist, device):
    # Arrange
    check = TrainTestPredictionDrift().add_condition_drift_score_not_greater_than()
    mod_train_ds, mod_test_ds = mnist_drifted_datasets

    def infer(batch, model, device):
        preds = model.to(device)(batch[0].to(device))
        preds[:, 0] = 0
        preds = nn.Softmax(dim=1)(preds)
        return preds

    mod_test_ds.infer_on_batch = infer

    # Act
    result = check.run(mod_train_ds, mod_test_ds, mock_trained_mnist, device=device, n_samples=None)

    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='PSI <= 0.15 and Earth Mover\'s Distance <= 0.075 for prediction drift',
        details='Found non-continues prediction properties with PSI drift score above threshold: {\'Samples Per '
                'Class\': \'4.0\'}\n'
    ))


def test_with_drift_object_detection_defected_alternative_measurements():
    # Arrange
    def prop(predictions):
        return [int(x[0][0]) if len(x) != 0 else 0 for x in predictions]
    alternative_measurements = [
        {'name': 'test', 'method': prop, 'output_type': 'continuous'},
        {'name234': 'test', 'method': prop, 'output_type': 'continuous'},
    ]

    # Assert
    assert_that(calling(TrainTestPredictionDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       r"Property must be of type dict, and include keys \['name', 'method', 'output_type'\]")
                )


def test_with_drift_object_detection_defected_alternative_measurements2():
    # Arrange
    alternative_measurements = {'name': 'test', 'method': lambda x: x, 'output_type': 'continuous'}

    # Assert
    assert_that(calling(TrainTestPredictionDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       "Expected properties to be a list, instead got dict")
                )
