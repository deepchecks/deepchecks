# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Test for the properties module"""
from unittest.mock import patch

import pytest
from hamcrest import assert_that, close_to, equal_to

from deepchecks.nlp.utils.text_properties import calculate_default_properties


def mock_fn(*args, **kwargs):  # pylint: disable=unused-argument
    return [0] * 20_000


@patch('deepchecks.nlp.utils.text_properties.run_available_kwargs', mock_fn)
def test_calculate_default_properties():
    # Arrange
    raw_text = ['This is a test sentence.'] * 20_000

    match_text = r'Calculating the properties \[\'Toxicity\'\] on a large dataset may take a long time.' \
                 r' Consider using a smaller sample size or running this code on better hardware. Consider using a ' \
                 r'GPU or a similar device to run these properties.'

    # Act
    with pytest.warns(UserWarning,
                      match=match_text):
        result = calculate_default_properties(raw_text, include_properties=['Toxicity'],
                                              include_long_calculation_properties=True)[0]

    # Assert
    assert_that(result, equal_to({'Toxicity': [0] * 20_000}))
