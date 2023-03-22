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
"""Test for the TextData object"""
import pandas as pd
from hamcrest import assert_that, calling, raises, equal_to, contains_exactly

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.text_data import TextData
from deepchecks.nlp.utils.text_properties import LONG_RUN_PROPERTIES


def test_text_data_init():
    """Test the TextData object initialization"""
    text_data = TextData(['Hello world'])  # should pass
    assert_that(text_data.text, contains_exactly('Hello world'))


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
        calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, r'label must be a Sequence of Sequences of either strings or integers')
    )

    # Arrange
    label = [['PER', 'ORG', 'ORG', 'GEO'], [], []]

    # Act & Assert
    assert_that(
        calling(TextData).with_args(raw_text=text, label=label, task_type='text_classification'),
        raises(DeepchecksValueError,
               r'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.')
    )


def test_wrong_token_label_format():
    # Arrange
    text = ['a', 'b b b', 'c c c c']

    label_structure_error = r'label must be a Sequence of Sequences of either strings or integers'

    # Act & Assert
    # OK sample:
    label = [['B-PER'],
             ['B-PER', 'B-GEO', 'B-GEO'],
             ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    _ = TextData(raw_text=text, label=label, task_type='token_classification')  # Should pass

    # Not a list:
    label = 'PER'
    assert_that(
        calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, 'label must be a Sequence')
    )

    # Not a list of lists:
    label = [3, 3, 3]
    assert_that(
        calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, label_structure_error)
    )

    # Mixed strings and integers:
    label = [['B-PER'],
             ['B-PER', 1, 'B-GEO'],
             ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    assert_that(
        calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, label_structure_error)
    )

    # Not of same length:
    label = [['B-PER'],
             ['B-PER', 'B-GEO', 'B-GEO'],
             ['B-PER', 'B-GEO', 'B-GEO']]
    assert_that(
        calling(TextData).with_args(raw_text=text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, r'label must be the same length as tokenized_text. '
                                     r'However, for sample index 2 of length 4 received label of length 3')
    )


def test_metadata_format():
    # Arrange
    text = ['a', 'b b b', 'c c c c']
    metadata = {'first': [1, 2, 3], 'second': [4, 5, 6]}

    # Act & Assert
    _ = TextData(raw_text=text, metadata=pd.DataFrame(metadata),
                 task_type='text_classification')  # Should pass
    assert_that(
        calling(TextData).with_args(raw_text=text, metadata=metadata, task_type='text_classification'),
        raises(DeepchecksValueError,
               r"metadata type <class 'dict'> is not supported, must be a pandas DataFrame")
    )


def test_head_functionality():
    # Arrange
    text = ['a', 'b b b', 'c c c c']
    metadata = {'first': [1, 2, 3], 'second': [4, 5, 6]}
    label = ['PER', 'ORG', 'GEO']

    # Act
    dataset = TextData(raw_text=text, metadata=pd.DataFrame(metadata),
                       task_type='text_classification', label=label)
    result = dataset.head(n_samples=2)

    # Assert
    assert_that(len(result), equal_to(2))
    assert_that(sorted(result.columns), contains_exactly('first', 'label', 'second', 'text'))
    assert_that(list(result.index), contains_exactly(0, 1))


def test_properties(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock

    # Act & Assert
    assert_that(dataset._properties, equal_to(None))  # pylint: disable=protected-access
    # TODO: Create test for the heavy properties
    dataset.calculate_default_properties(ignore_properties=['topic'] + LONG_RUN_PROPERTIES)
    properties = dataset.properties
    assert_that(properties.shape[0], equal_to(3))
    assert_that(properties.shape[1], equal_to(6))
    assert_that(properties.columns,
                contains_exactly('Text Length', 'Average Word Length', '% Special Characters', 'Language',
                                 'Sentiment', 'Subjectivity'))
    assert_that(properties.iloc[0].values, contains_exactly(22, 3.6, 0.0, 'en', 0.0, 0.0))
