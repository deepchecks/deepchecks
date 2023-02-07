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
"""Test for the default suites"""
from deepchecks.nlp.suites import full_suite
from tests.common import get_expected_results_length, validate_suite_result


def test_full_suite(movie_reviews_data):
    # Arrange
    train_data, test_data = movie_reviews_data
    args = dict(train_dataset=train_data, test_dataset=test_data)

    # Act
    suite = full_suite(imaginary_kwarg='just to make sure all checks have kwargs in the init')
    result = suite.run(**args)

    # Assert
    length = get_expected_results_length(suite, args)
    validate_suite_result(result, length)
