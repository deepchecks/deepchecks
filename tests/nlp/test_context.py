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
"""Test for the Context & _DummyModel creation process"""
from hamcrest import assert_that, calling, raises

from deepchecks.core.errors import ValidationError
from deepchecks.nlp import Suite


def test_wrong_class_prediction_format(text_classification_dataset_mock):

    # Arrange
    emtpy_suite = Suite('Empty Suite')

    # Act & Assert
    assert_that(calling(emtpy_suite.run).with_args(
        train_dataset=text_classification_dataset_mock,
        train_predictions=[0, 0, 1, 1]),
        raises(ValidationError, 'Check requires predictions for train to have 3 rows, same as dataset')
    )
