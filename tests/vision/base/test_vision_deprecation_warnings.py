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
"""Contains unit tests for the vision package deprecation warnings."""

import warnings

import pytest
import torch
from ignite.metrics import Accuracy

from deepchecks.vision.checks import (ClassPerformance, ImageSegmentPerformance, RobustnessReport,
                                      SimpleModelComparison, SingleDatasetPerformance)


def test_deprecation_warning_robustness_report():
    # Test that warning is raised when alternative_metrics has value:
    with pytest.warns(DeprecationWarning, match='alternative_metrics'):
        check = RobustnessReport(alternative_metrics={'ac': Accuracy()})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = RobustnessReport()


def test_deprecation_warning_simple_model_comparison():
    # Test that warning is raised when alternative_metrics has value:
    with pytest.warns(DeprecationWarning, match='alternative_metrics'):
        check = SimpleModelComparison(alternative_metrics={'ac': Accuracy()})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = SimpleModelComparison()


def test_deprecation_warning_image_segment_performance():
    # Test that warning is raised when alternative_metrics has value:
    with pytest.warns(DeprecationWarning, match='alternative_metrics'):
        check = ImageSegmentPerformance(alternative_metrics={'ac': Accuracy()})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = ImageSegmentPerformance()


def test_deprecation_warning_class_performance():
    # Test that warning is raised when alternative_metrics has value:
    with pytest.warns(DeprecationWarning, match='alternative_metrics'):
        check = ClassPerformance(alternative_metrics={'ac': Accuracy()})

    # Check to see no warnings are raised when deprecated feature doesn't exist:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check = ClassPerformance()


def test_deprecation_train_predictions(mnist_dataset_train):
    pred_train = torch.rand((mnist_dataset_train.num_samples, 10))
    pred_train = pred_train / torch.sum(pred_train, dim=1, keepdim=True)
    pred_train_dict = dict(zip(range(mnist_dataset_train.num_samples), pred_train))
    with pytest.warns(DeprecationWarning,
                      match='train_predictions is deprecated, please use predictions instead.'):
        _ = SingleDatasetPerformance().run(mnist_dataset_train, train_predictions=pred_train_dict)
