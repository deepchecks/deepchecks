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
import numpy as np

from deepchecks.vision import VisionDataset
from deepchecks.vision.utils import ClassificationLabelFormatter
from hamcrest import assert_that, instance_of, calling, raises, equal_to
from tests.vision.vision_conftest import *


def test_classification_formatter_invalid_dataloader(three_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(three_tuples_dataloader)
    assert_that(err, equal_to("Check requires dataloader to return tuples of (input, label)."))


def test_classification_formatter_formatting_valid_label_shape(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: x)

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to(None))


def test_classification_formatter_formatting_invalid_label_type(two_tuples_dataloader):
    formatter = ClassificationLabelFormatter(lambda x: [x,x])

    err = formatter.validate_label(two_tuples_dataloader)
    assert_that(err, equal_to("Check requires classification label to be a torch.Tensor or numpy array"))
