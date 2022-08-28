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
from deepchecks import CheckResult, ConditionResult, ConditionCategory
from deepchecks.core import DatasetKind
from deepchecks.utils.strings import format_number
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.drift import word_counts_drift_plot
from deepchecks.nlp.base_checks import TrainTestCheck
from deepchecks.utils.distribution.drift import cramers_v, psi
from typing import Union, List, Any
from deepchecks.nlp.context import Context
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np

__all__ = ['KeywordFrequencyDrift']


class KeywordFrequencyDrift(TrainTestCheck):
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

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        train_dataset = context.train
        test_dataset = context.test
        all_data = list(train_dataset.text) + list(test_dataset.text)

        vectorizer = TfidfVectorizer(input='content', strip_accents='ascii', preprocessor=self.stem_func,
                                     token_pattern=self.token_pattern)
        vectorizer.fit(all_data)
        train_freqs = vectorizer.transform(train_dataset.text)
        max_train_freqs = np.array(train_freqs.max(axis=0).todense()).reshape(-1)

        test_freqs = vectorizer.transform(test_dataset.text)
        max_test_freqs = np.array(test_freqs.max(axis=0).todense()).reshape(-1)
        word_freq_diff = np.abs(max_train_freqs - max_test_freqs)

        max_train_counts = max_train_freqs * test_dataset.n_samples
        max_test_counts = max_test_freqs * test_dataset.n_samples

        drift_score = self.drift_method(max_train_counts, max_test_counts, from_freqs=True)
        vocab = vectorizer.get_feature_names_out()

        if isinstance(self.top_n_method, List):
            top_n_idxs = [idx for idx, word in enumerate(vocab) if word in self.top_n_method]
        elif self.top_n_method == 'top_diff':
            top_n_idxs = np.argsort(word_freq_diff)[-self.top_n_to_show:]
        elif self.top_n_method == 'top_freq':
            max_freqs = np.maximum(max_train_freqs, max_test_freqs)
            top_n_idxs = np.argsort(max_freqs, )[-self.top_n_to_show:]
        else:
            raise DeepchecksValueError('top_n_method must be one of: top_diff, top_freq or a list of keywords')

        top_n_words = np.take(np.array(vocab), top_n_idxs)
        top_n_diffs = np.take(word_freq_diff, top_n_idxs)

        if context.with_display:
            train_to_show = max_train_freqs[top_n_idxs]
            test_to_show = max_test_freqs[top_n_idxs]
            display = word_counts_drift_plot(train_to_show, test_to_show, top_n_words)
        else:
            display = None

        result = {'drift_score': drift_score, 'top_n_diffs': dict(zip(top_n_words, top_n_diffs))}
        return CheckResult(value=result, display=display, header='Keyword Frequency Drift')


    def add_condition_drift_score_less_than(self, threshold: float):
        """
        Add condition - require drift score to be less than the threshold.
        """
        def condition(value) -> ConditionResult:
            drift_score = value['drift_score']
            if drift_score < threshold:
                details = f'The drift score {format_number(drift_score)} is less than the threshold {format_number(threshold)}'
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                details = f'The drift score {format_number(drift_score)} is not less than the threshold {format_number(threshold)}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Drift Score is Less Than {format_number(threshold)}', condition)

    def add_condition_top_n_differences_less_than(self, threshold: float):
        """
        Add condition - require the absolute differences between the counts of train and the test to be less than the
        threshold for all of the top n keywords.
        """
        def condition(value) -> ConditionResult:
            diffs = value['top_n_diffs']
            keywords_failed = [k for k,v in diffs.items() if v >= threshold]

            if len(keywords_failed) == 0:
                details = 'Passed for all of the top N keywords'
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                details = f'Failed for the keywords: {keywords_failed}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Diffrences between the frequencies of the top N keywords are less than '
                                  f'{format_number(threshold)}', condition)
