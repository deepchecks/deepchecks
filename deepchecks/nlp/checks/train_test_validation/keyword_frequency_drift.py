# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing the keyword drift check."""
from deepchecks import CheckResult
from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.base_checks import TrainTestBaseCheck
from deepchecks.utils.distribution.drift import cramers_v, psi
from typing import Union, List, Any
from deepchecks.nlp.context import Context
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np

__all__ = ['KeywordFrequencyDrift']


class KeywordFrequencyDrift(TrainTestBaseCheck):
    """
    Computes the keywords' frequencies drift between the train and the test datasets.

    Parameters
    ----------
    top_n_to_show: int, default: 20
        How many words will be displayed in the graph.
    top_n_method: Union[str, List[str]], default: 'top_diff'
        Decides which method will be used to select the top n words to show. Possible values are:
        "top_diff" for the top difference between train and test, "top_freq" for the top absolute frequencies, or a list
        of keywords in which case the words on the list are shown and the "top_n_to_show" parameter is ignored.
    drift_method: str, default: "cramer_v"
        Decides which method will be used for drift calculation. Possible values are:
        "cramer_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    """

    def __init__(self,
                 top_n_to_show: int = 20,
                 top_n_method: Union[str, List[str]] = 'top_diff',
                 drift_method='cramer_v',
                 **kwargs):
        super().__init__(**kwargs)
        self.top_n_to_show = top_n_to_show
        self.top_n_method = top_n_method

        if drift_method == 'PSI':
            self.drift_method = psi
        elif drift_method == 'cramer_v':
            self.drift_method = cramers_v
        else:
            raise DeepchecksValueError('drift_method must be one of: PSI, cramer_v')
        self.stem_func = PorterStemmer().stem
        self.token_pattern = r'[a-z]{2,}'

        self.top_n_words = None
        self.top_n_diffs = None

    def run(self, train_dataset, test_dataset, model=None, **kwargs) -> CheckResult:
        """Run check."""
        all_data = list(train_dataset.text) + list(test_dataset.text)

        vectorizer = TfidfVectorizer(input='content', strip_accents='ascii', preprocessor=self.stem_func,
                                     token_pattern=self.token_pattern)
        vectorizer.fit(all_data)
        train_freqs = vectorizer.transform(train_dataset.text)
        mean_train_freqs = np.array(np.mean(train_freqs, axis=0)).reshape(-1)
        test_freqs = vectorizer.transform(test_dataset.text)
        mean_test_freqs = np.array(np.mean(test_freqs, axis=0)).reshape(-1)
        word_freq_diff = np.abs(mean_train_freqs - mean_test_freqs)

        drift_score = self.drift_method(mean_test_freqs, mean_train_freqs, from_freqs=True)
        vocab = vectorizer.get_feature_names_out()

        if isinstance(self.top_n_method, List):
            top_n_idxs = [idx for word, idx in enumerate(vocab) if word in self.top_n_method]
        elif self.top_n_method == 'top_diff':
            top_n_idxs = np.argsort(word_freq_diff)[:self.top_n_to_show]
        elif self.top_n_method == 'top_freq':
            max_freqs = np.maximum(mean_train_freqs, mean_test_freqs)
            top_n_idxs = np.argsort(max_freqs)[:self.top_n_to_show]
        else:
            raise DeepchecksValueError('top_n_method must be one of: top_diff, top_freq or a list of keywords')

        self.top_n_words = np.take(np.array(vocab), top_n_idxs)
        self.top_n_diffs = np.take(word_freq_diff, top_n_idxs)

        return CheckResult(drift_score, header='Keyword Frequency Drift')
