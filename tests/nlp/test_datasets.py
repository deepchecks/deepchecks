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

import numpy as np
from hamcrest import assert_that, contains_exactly, equal_to

from deepchecks.nlp.datasets.classification import just_dance_comment_analysis, tweet_emotion
from deepchecks.nlp.datasets.token_classification import scierc_ner


def test_tweet_emotion():
    # Arrange
    train, test = tweet_emotion.load_data(data_format='Dataframe', as_train_test=True)
    full = tweet_emotion.load_data(data_format='Dataframe', as_train_test=False)
    full_ds = tweet_emotion.load_data(data_format='TextData', as_train_test=False)
    preds = tweet_emotion.load_precalculated_predictions(pred_format='predictions', as_train_test=False)
    probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities', as_train_test=False)
    properties = tweet_emotion.load_properties(as_train_test=False)
    train_props, test_props = tweet_emotion.load_properties(as_train_test=True)
    embeddings = tweet_emotion.load_embeddings(as_train_test=False)
    train_embeddings, test_embeddings = tweet_emotion.load_embeddings(as_train_test=True)

    # Act & Assert
    assert_that(len(train) + len(test), equal_to(len(full)))
    assert_that(train.columns, contains_exactly(*test.columns))
    assert_that(train.columns, contains_exactly(*full.columns))

    assert_that(len(full_ds.text), equal_to(len(full)))
    assert_that(len(full.text), equal_to(len(preds)))
    assert_that([tweet_emotion._LABEL_MAP[x] for x in np.argmax(probas, axis=1)],  # pylint: disable=protected-access
                contains_exactly(*preds))

    assert_that(len(properties), equal_to(len(full)))
    assert_that(len(train_props) + len(test_props), equal_to(len(full)))
    assert_that(len(train_props), equal_to(len(train)))

    assert_that(len(embeddings), equal_to(len(full)))
    assert_that(len(train_embeddings) + len(test_embeddings), equal_to(len(full)))
    assert_that(len(train_embeddings), equal_to(len(train)))
    assert_that(embeddings.shape, contains_exactly(4653, 1536))
    assert_that(train_embeddings.shape, contains_exactly(2675, 1536))
    assert_that(test_embeddings.shape, contains_exactly(1978, 1536))


def test_just_dance_comment_analysis():
    # Arrange
    train, test = just_dance_comment_analysis.load_data(data_format='Dataframe', as_train_test=True)
    full = just_dance_comment_analysis.load_data(data_format='Dataframe', as_train_test=False)
    full_ds = just_dance_comment_analysis.load_data(data_format='TextData', as_train_test=False,
                                                    include_embeddings=True)
    preds = just_dance_comment_analysis.load_precalculated_predictions(pred_format='predictions', as_train_test=False)
    probas = just_dance_comment_analysis.load_precalculated_predictions(pred_format='probabilities',
                                                                        as_train_test=False)
    properties = just_dance_comment_analysis.load_properties(as_train_test=False)
    train_props, test_props = just_dance_comment_analysis.load_properties(as_train_test=True)
    embeddings = just_dance_comment_analysis.load_embeddings(as_train_test=False)
    train_embeddings, test_embeddings = just_dance_comment_analysis.load_embeddings(as_train_test=True)

    # Act & Assert
    assert_that(len(train) + len(test), equal_to(len(full)))
    assert_that(train.columns, contains_exactly(*test.columns))
    assert_that(train.columns, contains_exactly(*full.columns))

    assert_that(len(full_ds.text), equal_to(len(full)))
    assert_that(len(full_ds.text), equal_to(len(preds)))
    assert_that(len(full_ds.text), equal_to(len(probas)))

    assert_that(len(properties), equal_to(len(full)))
    assert_that(len(train_props) + len(test_props), equal_to(len(full)))
    assert_that(len(train_props), equal_to(len(train)))

    assert_that(len(embeddings), equal_to(len(full)))
    assert_that(len(train_embeddings) + len(test_embeddings), equal_to(len(full)))
    assert_that(len(train_embeddings), equal_to(len(train)))
    assert_that(embeddings.shape, contains_exactly(16281, 1536))
    assert_that(train_embeddings.shape, contains_exactly(7669, 1536))
    assert_that(test_embeddings.shape, contains_exactly(8612, 1536))


def test_scierc_ner_tokens():
    train_dict, test_dict = scierc_ner.load_data(data_format='Dict')
    train_ds, test_ds = scierc_ner.load_data(data_format='TextData')
    train_preds, test_preds = scierc_ner.load_precalculated_predictions()
    train_props, test_props = scierc_ner.load_properties()
    train_embeddings, test_embeddings = scierc_ner.load_embeddings()

    assert_that(len(train_dict['text']), equal_to(len(train_dict['label'])))
    assert_that(len(test_dict['text']), equal_to(len(test_dict['label'])))

    assert_that(len(train_ds.text), equal_to(len(train_preds)))
    assert_that(len(test_ds.text), equal_to(len(test_preds)))

    assert_that(train_props.columns, contains_exactly(*test_props.columns))

    assert_that(train_embeddings.shape, contains_exactly(350, 1536))
    assert_that(test_embeddings.shape, contains_exactly(100, 1536))
