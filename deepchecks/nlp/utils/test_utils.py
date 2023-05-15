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
"""utils for testing."""

from deepchecks.nlp.datasets.classification import tweet_emotion


def load_modified_tweet_text_data():
    """Load tweet emotion data and modify the label of some samples."""
    text_data = tweet_emotion.load_data(as_train_test=False).copy()

    idx_to_change_metadata = list(text_data.metadata[(text_data.metadata['user_age'] > 40) & (
            text_data.metadata['user_region'] == 'Europe') & (text_data.metadata['user_age'] < 57)].index)

    idx_to_change_properties = list(text_data.properties[(text_data.properties['Formality'] > 0.4) & (
            text_data.properties['Text Length'] > 80) & (text_data.properties['Text Length'] < 130)].index)

    label = text_data._label.astype(object)  # pylint: disable=protected-access
    label[idx_to_change_metadata[int(len(idx_to_change_metadata) / 2):]] = None
    label[idx_to_change_properties[:int(len(idx_to_change_properties) / 2)]] = None

    text_data._label = label  # pylint: disable=protected-access

    return text_data
