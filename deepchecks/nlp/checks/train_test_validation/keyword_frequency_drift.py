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
import re
from collections import OrderedDict
from typing import List, Union

import numpy as np
import sklearn
from nltk import download as nltk_download
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from deepchecks import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.base_checks import TrainTestCheck
from deepchecks.nlp.context import Context
from deepchecks.utils.distribution.drift import cramers_v, psi, word_counts_drift_plot
from deepchecks.utils.strings import format_number

__all__ = ['KeywordFrequencyDrift']


class KeywordFrequencyDrift(TrainTestCheck):
    """
    Computes the keywords' frequencies drift between the train and the test datasets.

    Drift is a change in the distribution of the data over time. In this check, we look at the distribution of the
    keywords' TF-IDF scores.
    For more information about TF-IDF see https://en.wikipedia.org/wiki/Tf%E2%80%93idf.

    To calculate the drift score we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.

    Parameters
    ----------
    top_n_to_show: int, default: 20
        How many words will be displayed in the graph.
    top_n_method: Union[str, List[str]], default: 'top_diff'
        Decides which method will be used to select the top n words to show. Possible values:
        - 'top_diff': Show the words with the largest difference between train and test.
        - 'top_freq': Show the words with the largest absolute frequencies,
        - A list of keywords in which case the words on the list are shown and the "top_n_to_show" parameter is ignored.
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
        self.stemming_lookup = {}

        if drift_method == 'PSI':
            self.drift_method = psi
        elif drift_method == 'cramer_v':
            self.drift_method = cramers_v
        else:
            raise DeepchecksValueError(f'drift_method must be one of: PSI, cramer_v, found {drift_method}')
        nltk_download('punkt', quiet=True)
        nltk_download('stopwords', quiet=True)
        self.stopword_list = stopwords.words('english')
        self.token_pattern = r'[a-z]{2,}'

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train_dataset = context.train
        test_dataset = context.test

        tokenized_train = [self._tokenize(x) for x in train_dataset.text]
        tokenized_test = [self._tokenize(x) for x in test_dataset.text]
        all_data = tokenized_train + tokenized_test

        vectorizer = TfidfVectorizer(input='content', strip_accents='ascii', tokenizer=_identity_tokenizer, min_df=2,
                                     preprocessor=_identity_tokenizer, binary=True, stop_words=self.stopword_list)
        vectorizer.fit(all_data)
        train_freqs = vectorizer.transform(tokenized_train)
        max_train_freqs = np.array(train_freqs.max(axis=0).todense()).reshape(-1)

        test_freqs = vectorizer.transform(tokenized_test)
        max_test_freqs = np.array(test_freqs.max(axis=0).todense()).reshape(-1)
        word_freq_diff = max_train_freqs - max_test_freqs

        max_train_counts = max_train_freqs * train_dataset.n_samples
        max_test_counts = max_test_freqs * test_dataset.n_samples

        drift_score = self.drift_method(max_train_counts, max_test_counts, from_freqs=True)
        if int(sklearn.__version__.split('.', maxsplit=1)[0]) >= 1:
            vocab = vectorizer.get_feature_names_out()
        else:
            vocab = vectorizer.get_feature_names()

        if isinstance(self.top_n_method, List):
            top_n_idxs = [idx for idx, word in enumerate(vocab) if word in self.top_n_method]
        elif self.top_n_method == 'top_diff':
            top_n_idxs = np.argsort(np.abs(word_freq_diff))[-self.top_n_to_show:]
        elif self.top_n_method == 'top_freq':
            max_freqs = np.maximum(max_train_freqs, max_test_freqs)
            top_n_idxs = np.argsort(max_freqs, )[-self.top_n_to_show:]
        else:
            raise DeepchecksValueError('top_n_method must be one of: top_diff, top_freq or a list of keywords')

        top_n_stems = np.take(np.array(vocab), top_n_idxs)
        top_n_diffs = np.take(word_freq_diff, top_n_idxs)

        self._sort_lookup_by_value()
        top_n_words = [self._unstem(s) for s in top_n_stems]

        if context.with_display:
            dataset_names = (train_dataset.name, test_dataset.name)
            headnote = f"""<span>
                    The Drift score is a measure for the difference between two distributions, in this check - the
                    {dataset_names[0]} and {dataset_names[1]} distributions.<br> The check shows the differences between
                    the TF-IDF scores for the keywords in the {dataset_names[0]} and {dataset_names[1]} datasets.
                    </span>"""
            train_to_show = max_train_freqs[top_n_idxs]
            test_to_show = max_test_freqs[top_n_idxs]
            display = [headnote, word_counts_drift_plot(train_to_show, test_to_show, top_n_words, dataset_names)]
        else:
            display = None

        result = {'drift_score': drift_score, 'top_n_diffs': dict(zip(top_n_words, top_n_diffs))}
        return CheckResult(value=result, display=display, header='Keyword Frequency Drift')

    def add_condition_drift_score_less_than(self, threshold: float = 0.2):
        """Add condition - require drift score to be less than the threshold."""
        def condition(value) -> ConditionResult:
            drift_score = value['drift_score']
            if drift_score < threshold:
                details = f'Found drift score {format_number(drift_score)} under threshold ' \
                          f'{format_number(threshold)}'
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                details = f'Found drift score {format_number(drift_score)} above threshold ' \
                          f'{format_number(threshold)}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Drift Score is Less Than {format_number(threshold)}', condition)

    def add_condition_top_n_differences_less_than(self, threshold: float):
        """Add condition - require no big change in the frequency of top keywords.

        The condition requires that the absolute differences between the counts of train and the test to be less than
        the threshold for all the top n keywords.
        """
        def condition(value) -> ConditionResult:
            diffs = value['top_n_diffs']
            keywords_failed = [k for k, v in diffs.items() if abs(v) >= threshold]

            if len(keywords_failed) == 0:
                details = 'Passed for all of the top N keywords'
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                details = f'Failed for the keywords: {keywords_failed}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Differences between the frequencies of the top N keywords are less than '
                                  f'{format_number(threshold)}', condition)

    def _stem(self, word):
        """Stem and cache a word."""
        if word not in self.stemming_lookup:
            self.stemming_lookup[word] = LancasterStemmer().stem(word)
        return self.stemming_lookup[word]

    def _tokenize(self, text):
        """Tokenize text."""
        tokens = word_tokenize(text)
        stems = [self._stem(item) for item in tokens if re.match(self.token_pattern, item)]
        return stems

    def _unstem(self, stem):
        """Transform a stem into a readable word."""
        word = list(self.stemming_lookup)[list(self.stemming_lookup.values()).index(stem)]
        return word

    def _sort_lookup_by_value(self):
        sorted_lookup = OrderedDict(sorted(self.stemming_lookup.items(), key=lambda x: x[0]))
        self.stemming_lookup = sorted_lookup


def _identity_tokenizer(text):
    return text
