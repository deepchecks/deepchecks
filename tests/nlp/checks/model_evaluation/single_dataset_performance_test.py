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

from hamcrest import assert_that, close_to, equal_to, has_length

from deepchecks.nlp.checks.model_evaluation.single_dataset_performance import SingleDatasetPerformance


def test_run_with_scorer(text_classification_dataset_mock):
    """Test that the check runs with a scorer override"""
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_macro'])

    # Act
    result = check.run(text_classification_dataset_mock,
                       predictions=[0, 1, 1])

    # Assert
    assert_that(result.value.values[0][-1], close_to(0.666, 0.001))
