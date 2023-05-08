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
# pylint: disable=protected-access
"""Test for the TextData object"""
import pandas as pd
from hamcrest import assert_that, calling, contains_exactly, equal_to, raises

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
        raises(DeepchecksValueError, r'tokenized_text must be provided for token_classification task type')
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
    tokenized_text = [['a'] ,['b', 'b' ,'b'], ['c', 'c', 'c', 'c']]

    label_structure_error = r'label must be a Sequence of Sequences of either strings or integers'

    # Act & Assert
    # OK sample:
    label = [['B-PER'],
             ['B-PER', 'B-GEO', 'B-GEO'],
             ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    _ = TextData(tokenized_text=tokenized_text, label=label, task_type='token_classification')  # Should pass

    # Not a list:
    label = 'PER'
    assert_that(
        calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, 'label must be a Sequence')
    )

    # Not a list of lists:
    label = [3, 3, 3]
    assert_that(
        calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, label_structure_error)
    )

    # Mixed strings and integers:
    label = [['B-PER'],
             1,
             ['B-PER', 'B-GEO', 'B-GEO', 'B-GEO']]
    assert_that(
        calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, label_structure_error)
    )

    # Not of same length:
    label = [['B-PER'],
             ['B-PER', 'B-GEO', 'B-GEO'],
             ['B-PER', 'B-GEO', 'B-GEO']]
    assert_that(
        calling(TextData).with_args(tokenized_text=tokenized_text, label=label, task_type='token_classification'),
        raises(DeepchecksValueError, r'label must be the same length as tokenized_text. However, for sample '
                                     r'index 2 received token list of length 4 and label list of length 3')
    )


def test_text_data_initialization_with_incorrect_type_of_metadata():
    # Arrange
    text = ['a', 'b b b', 'c c c c']
    metadata = {'first': [1, 2, 3], 'second': [4, 5, 6]}

    # Act & Assert
    _ = TextData(
        raw_text=text,
        metadata=pd.DataFrame(metadata),
        task_type='text_classification'
    )
    assert_that(
        calling(TextData).with_args(
            raw_text=text,
            metadata=metadata,
            task_type='text_classification'
        ),
        raises(
            DeepchecksValueError,
            r"Metadata type <class 'dict'> is not supported, must be a pandas DataFrame"
        )
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
    assert_that(dataset._properties, equal_to(None))
    # TODO: Create test for the heavy properties
    dataset.calculate_default_properties(ignore_properties=['topic'] + LONG_RUN_PROPERTIES)
    properties = dataset.properties
    assert_that(properties.shape[0], equal_to(3))
    assert_that(properties.shape[1], equal_to(7))
    assert_that(properties.columns,
                contains_exactly('Text Length', 'Average Word Length', 'Max Word Length', '% Special Characters',
                                 'Sentiment', 'Subjectivity', 'Lexical Density'))
    assert_that(properties.iloc[0].values, contains_exactly(22, 3.6, 9, 0.0, 0.0, 0.0, 80.0 ))


def test_embeddings():
    ds = TextData(['my name is inigo montoya', 'you killed my father', 'prepare to die'])
    ds.calculate_default_embeddings()
    assert_that(ds.embeddings.shape, equal_to((3, 384)))


def test_set_embeddings(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    embeddings = pd.DataFrame({'0': [1, 2, 3], '1': [4, 5, 6]})
    assert_that(dataset._embeddings, equal_to(None))  # pylint: disable=protected-access

    dataset.set_embeddings(embeddings)
    assert_that((dataset.embeddings != embeddings).sum().sum(), equal_to(0))


def test_set_metadata(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})

    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))

    # Act
    dataset.set_metadata(metadata, categorical_metadata=[])

    # Assert
    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))
    assert_that(dataset.categorical_metadata_columns, equal_to([]))


def test_set_metadata_with_categorical_columns(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})

    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))

    # Act
    dataset.set_metadata(metadata, categorical_metadata=['second'])

    # Assert
    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))
    assert_that(dataset.categorical_metadata_columns, equal_to(['second']))


def test_set_metadata_with_an_incorrect_list_of_categorical_columns(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})

    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))

    # Act/Assert
    assert_that(
        calling(dataset.set_metadata).with_args(
            metadata,
            categorical_metadata=['foo']
        ),
        raises(
            DeepchecksValueError,
            r"The following columns does not exist in Metadata - \['foo'\]"
        )
    )

def test_load_metadata(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    metadata = pd.DataFrame({'first': [1, 2, 3], 'second': [4, 5, 6]})

    assert_that(dataset._metadata, equal_to(None))
    assert_that(dataset._cat_metadata, equal_to(None))

    metadata.to_csv('metadata.csv', index=False)
    loaded_metadata = pd.read_csv('metadata.csv')

    assert_that((loaded_metadata != metadata).sum().sum(), equal_to(0))

    dataset.set_metadata(loaded_metadata)

    assert_that((dataset.metadata != metadata).sum().sum(), equal_to(0))


def test_set_properties(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})

    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))

    # Act
    dataset.set_properties(properties, categorical_properties=[])

    # Assert
    assert_that(dataset.categorical_properties, equal_to([]))
    assert_that((dataset.properties != properties).sum().sum(), equal_to(0))

    dataset._properties = None
    dataset._cat_properties = None


def test_set_properties_with_an_incorrect_list_of_categorical_columns(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})

    # Act/Assert
    assert_that(
        calling(dataset.set_properties).with_args(
            properties,
            categorical_properties=['foo']
        ),
        raises(
            DeepchecksValueError,
            r"The following columns does not exist in Properties - \['foo'\]"
        )
    )


def test_set_properties_with_categorical_columns(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'unknown_property': ['foo', 'foo', 'bar']})

    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))

    # Act
    dataset.set_properties(properties)

    # Assert
    assert_that(dataset.categorical_properties, equal_to(['unknown_property']))


def test_save_and_load_properties(text_classification_dataset_mock):
    # Arrange
    dataset = text_classification_dataset_mock
    properties = pd.DataFrame({'text_length': [1, 2, 3], 'average_word_length': [4, 5, 6]})

    assert_that(dataset._properties, equal_to(None))
    assert_that(dataset._cat_properties, equal_to(None))

    # Act
    dataset.set_properties(properties, categorical_properties=[])
    dataset.save_properties('test_properties.csv')

    # Make sure is saved correctly:

    properties_loaded = pd.read_csv('test_properties.csv')

    assert_that((properties_loaded != properties).sum().sum(), equal_to(0))

    # Load into the dataset:

    dataset._properties = None
    dataset.set_properties('test_properties.csv')

    assert_that((dataset.properties != properties).sum().sum(), equal_to(0))
