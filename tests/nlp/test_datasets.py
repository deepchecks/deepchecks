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
from hamcrest import assert_that, equal_to, contains_exactly

from deepchecks.nlp.datasets.classification import tweet_emotion


def test_tweet_emotion():
    # Arrange
    train, test = tweet_emotion.load_data(data_format='Dataframe', as_train_test=True)
    full = tweet_emotion.load_data(data_format='Dataframe', as_train_test=False)
    full_ds = tweet_emotion.load_data(data_format='TextData', as_train_test=False)
    preds = tweet_emotion.load_precalculated_predictions(pred_format='predictions')
    probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities')
    properties = tweet_emotion.load_properties(as_train_test=False)
    train_props, test_props = tweet_emotion.load_properties(as_train_test=True)

    # Act & Assert
    assert_that(len(train) + len(test), equal_to(len(full)))
    assert_that(train.columns, contains_exactly(*test.columns))
    assert_that(train.columns, contains_exactly(*full.columns))

    assert_that(len(full_ds.text), equal_to(len(full)))
    assert_that(len(full.text), equal_to(len(preds)))
    assert_that([tweet_emotion._LABEL_MAP[x] for x in np.argmax(probas, axis=1)], contains_exactly(*preds))

    assert_that(len(properties), equal_to(len(full)))
    assert_that(len(train_props) + len(test_props), equal_to(len(full)))
    assert_that(len(train_props), equal_to(len(train)))
