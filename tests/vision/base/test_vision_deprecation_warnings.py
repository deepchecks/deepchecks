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

import pytest
import torch
from deepchecks.vision.checks import SingleDatasetPerformance


def test_deprecation_train_predictions(mnist_dataset_train):
    pred_train = torch.rand((mnist_dataset_train.num_samples, 10))
    pred_train = pred_train / torch.sum(pred_train, dim=1, keepdim=True)
    pred_train_dict = dict(zip(range(mnist_dataset_train.num_samples), pred_train))
    with pytest.warns(DeprecationWarning,
                      match='train_predictions is deprecated, please use predictions instead.'):
        _ = SingleDatasetPerformance().run(mnist_dataset_train, train_predictions=pred_train_dict)
