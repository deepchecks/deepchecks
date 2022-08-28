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
"""Test for the TextData object"""
from hamcrest import assert_that, calling, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.text_data import TextData


def test_text_data_init():
    """Test the TextData object initialization"""
    text_data = TextData(['Hello world'])
    assert_that(text_data.text, calling(str))
    assert_that(text_data.text, 'Hello world')


def test_init_no_text():
    """Test the TextData object when no text is provided"""

    # Act & Assert
    assert_that(
        calling(TextData).with_args([1]),
        raises(DeepchecksValueError, 'raw_text must be a Sequence of strings')
    )


def test_init_mismatched_task_type():
    """Test the TextData object when the task type does not match the label format"""

    # Arrange
    label = [1, 2, 3]
    text = ['a', 'b', 'c']

    # Act & Assert
    assert_that(
        calling(TextData).with_args(text, label, task_type='token_classification'),
        raises(DeepchecksValueError,
               r'label must be a Sequence of Sequences of \(str, int, int\) tuples, where the string is the token '
               r'label, the first int is the start of the token span in the raw text and the second int is the end of '
               r'the token span.')
    )

    # Arrange
    label = [[('PER', 3, 5), ('ORG', 5, 7)], [('ORG', 13, 15), ('GEO', 23, 25)], []]

    # Act & Assert
    assert_that(
        calling(TextData).with_args(text, label, task_type='text_classification'),
        raises(DeepchecksValueError,
               r'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.')
    )
