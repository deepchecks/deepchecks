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
"""Utils module containing scorers for computing token classification metrics."""
import typing as t
from collections.abc import Sequence

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import Token
from sklearn.metrics import make_scorer

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.text_data import TTokenLabel

__all__ = ['get_default_token_scorers', 'validate_scorers', 'get_scorer_dict']

DEFAULT_AVG_SCORER_NAMES = ('f1_macro', 'recall_macro', 'precision_macro')
DEFAULT_PER_CLASS_SCORER_NAMES = ('f1_per_class', 'f1_per_class', 'f1_per_class')


TSpanAligner = t.TypeVar('TSpanAligner', bound='SpanAligner')
if t.TYPE_CHECKING:
    from deepchecks.nlp.context import TTokenPred  # pylint: disable=unused-import # noqa: F401


def get_scorer_dict(span_aligner: TSpanAligner,
                    suffix: bool = False, mode: t.Optional[str] = None, scheme: t.Optional[t.Type[Token]] = None,
                    ) -> t.Dict[str, t.Callable[[t.List[str], t.List[str]], float]]:
    """Return a dict of scorers for token classification.

    Parameters:
    -----------
        mode: str, [None (default), `strict`].
            if ``None``, the score is compatible with conlleval.pl. Otherwise,
            the score is calculated strictly.
        scheme: Token, [IOB2, IOE2, IOBES]
        suffix: bool, False by default.

    Returns:
    --------
        A dict of scorers.
    """
    common_kwargs = {
        'span_aligner': span_aligner,
        'mode': mode,
        'scheme': scheme,
        'suffix': suffix,
        'zero_division': 0,
    }

    return {
        'token_accuracy': make_token_scorer(accuracy_score, span_aligner=span_aligner),
        'token_f1_per_class': make_token_scorer(f1_score, **common_kwargs, average=None),
        'token_f1_macro': make_token_scorer(f1_score, **common_kwargs, average='macro'),
        'token_f1_micro': make_token_scorer(f1_score, **common_kwargs, average='micro'),
        'token_precision_per_class': make_token_scorer(precision_score, **common_kwargs, average=None),
        'token_precision_macro': make_token_scorer(precision_score, **common_kwargs, average='macro'),
        'token_precision_micro': make_token_scorer(precision_score, **common_kwargs, average='micro'),
        'token_recall_per_class': make_token_scorer(recall_score, **common_kwargs, average=None),
        'token_recall_macro': make_token_scorer(recall_score, **common_kwargs, average='macro'),
        'token_recall_micro': make_token_scorer(recall_score, **common_kwargs, average='micro'),
    }


class SpanAligner:
    """An object for transforming the labels and predictions of a specific dataset.

    Labels come in the deepchecks accepted format and are returned as a list of lists of string annotations, with each
    inner list representing a sentence and containing one annotation per token.

    Examples:
    ---------
    A simple example for the output annotation format created by SpanAligner::

        text = ['Dan crossed the road', 'I live in Britain']
        label = [['PER', 'O', 'O', 'O'], ['O', 'O', 'O', 'GEO']]
        Pred = [['PER', 'O', 'O', 'GEO'], ['O', 'O', 'O', 'GEO']]
    """

    def __init__(self, strict_alignment: bool = True, proba_threshold: float = 0.):
        """Initialize the SpanAligner object.

        Parameters:
        -----------
        strict_alignment: bool, default: True
            If true, predictions are considered only if their spans match exactly any span in the labels. If false,
            predictions are considered if they overlap any label span.
        proba_threshold: float, default: 0
            The minimal probability score for a prediction to be considered.
        """
        if not isinstance(strict_alignment, bool):
            raise TypeError('strict_alignment must be a boolean')
        if not strict_alignment:
            # TODO: Implement non-strict alignment, probably using https://github.com/biocore-ntnu/ncls or similar
            raise NotImplementedError('Non Strict alignment is not yet implemented')
        if (not isinstance(proba_threshold, float)) or (proba_threshold < 0) or (proba_threshold > 1):
            raise TypeError('proba_threshold must be a float between 0 and 1')

        self.strict_alignment = strict_alignment
        self.proba_threshold = proba_threshold
        self.processed_labels = None
        self.processed_predictions = None
        self.classes = set()
        self._label_hash = 0
        self._pred_hash = 0

    @staticmethod
    def _hash_token_annotations(token_annot: TTokenLabel) -> int:
        """Hash token annotations to a unique integer."""
        return hash(tuple(map(tuple, token_annot)))

    def fit(self, y_true: TTokenLabel, y_pred: 'TTokenPred'):
        """Fit the SpanAligner object to the labels and predictions of a specific dataset."""
        self.classes = set()
        self.processed_labels, self.processed_predictions = self._spans_to_token_list(y_true, y_pred)
        self._label_hash = self._hash_token_annotations(y_true)
        self._pred_hash = self._hash_token_annotations(y_pred)

        # Post process class list to match the class order received by seqeval metrics
        self.classes.discard('O')
        self.classes = sorted(list(self.classes))

    def get_transformed(self) -> t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]:
        """Get the precomputed transformed labels and predictions."""
        return self.processed_labels, self.processed_predictions

    def fit_transform(self, y_true: TTokenLabel, y_pred: 'TTokenPred'
                      ) -> t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]:
        """Fit (if necessary) and transform the labels and predictions of a specific dataset."""
        # If fit was called in the past on the same labels and predictions, can skip fitting
        if (self._pred_hash != self._hash_token_annotations(y_pred)) or \
                (self._label_hash != self._hash_token_annotations(y_true)):
            self.fit(y_true, y_pred)
        return self.get_transformed()

    def _spans_to_token_list(self, y_true: TTokenLabel, y_pred: 'TTokenPred',
                             ) -> t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]:
        """Convert from predictions and labels in span format to a list of token annotations.

        Parameters:
        -----------
            y_true: TTokenLabel
                Labels as provided by the user in the accepted deepchecks format (TTokenLabel, detailed in TextData)
            y_pred: TTokenPred
                Predictions as provided by the user in the accepted format (TTokenPred, detailed in _shared_docs.py)

        Returns:
        --------
            A tuple of processed labels and processed predictions, each a list of lists of string annotations, as
            detailed on the class docstring.
        """
        if not len(y_true) == len(y_pred):
            raise DeepchecksValueError(f'Must have both a label entry and a prediction entry for each sample, but '
                                       f'found {len(y_true)} label entries and {len(y_pred)} prediction entries')

        labels = []
        predictions = []
        for sample_idx in range(len(y_true)):
            # Sort label and predictions by the starting position of the span
            sample_y_true = sorted(y_true[sample_idx], key=lambda x: x[1])
            sample_y_pred = sorted(y_pred[sample_idx], key=lambda x: x[1])
            labels.append([annotation[0] for annotation in sample_y_true])

            # A list of the spans, sorted by their starting position
            label_spans = [(annotation[1], annotation[2]) for annotation in sample_y_true]
            pred_spans = [(annotation[1], annotation[2]) for annotation in sample_y_pred]

            # Here we want to find the prediction spans that match specific label spans, and discard the rest.
            # We want to do that in O(N*log(N)) - assuming both lists are of length N approximately. Thus, the use of
            # np.searchsorted
            if len(sample_y_true) > 0:
                # Indices of where to place elements of label_spans in pred_spans so that pred_spans is still sorted.
                sorted_label_in_pred_index = np.searchsorted(list(map(lambda x: x[0], pred_spans)),
                                                             list(map(lambda x: x[0], label_spans)), side='left')
                # Take the indices of the elements of pred_spans that are currently in these positions, meaning
                # the prospective matches for each of the elements of label_spans
                possible_matching_pred_index = np.take(np.arange(len(pred_spans)),
                                                       sorted_label_in_pred_index, mode='clip')
                # Create a boolean array, telling us for each label_spans element whether it's prospective match in
                # pred_spans are identical in span (start *and* end position), and in the token class itself.
                bool_matching = np.array(list(map(hash, pred_spans)))[possible_matching_pred_index] == \
                    np.array(list(map(hash, label_spans)))
                # Filter out predictions that do not match their corresponding labels or their threshold is too low.
                # Predictions that where filtered are filled with 'O'.
                sample_predictions = \
                    [sample_y_pred[possible_matching_pred_index[idx]][0] if
                     ((sample_y_pred[possible_matching_pred_index[idx]][3] >= self.proba_threshold)
                      and bool_matching[idx]) else 'O'
                     for idx in range(len(sample_y_true))]
            else:
                sample_predictions = []

            self.classes.update(labels[-1])
            self.classes.update(sample_predictions)

            predictions.append(sample_predictions)

        return labels, predictions


def get_tokens(token_annotations: t.Sequence[t.Sequence[t.Tuple[str, int, int, t.Any]]]) -> t.Set[str]:
    """Return the token strings from token classification labels or predictions."""
    tokens = set()
    for sample_annotations in token_annotations:
        for annotation in sample_annotations:
            tokens.update(annotation[0])
    return tokens


def make_modified_metric(metric: t.Callable[[t.List[t.List[str]], t.List[t.List[str]], t.Any], float],
                         span_aligner: SpanAligner, **kwargs) -> t.Callable:
    """Apply spans_to_token_list processing to the labels and predictions, and then pass them to the metric function."""

    def modified_metric(y_true: TTokenLabel, y_pred: 'TTokenPred'):
        y_true_processed, y_pred_processed = span_aligner.fit_transform(y_true, y_pred, )
        return metric(y_true_processed, y_pred_processed, **kwargs)

    return modified_metric


def make_token_scorer(metric: t.Callable[[t.List[t.List[str]], t.List[t.List[str]]], float],
                      span_aligner: SpanAligner,
                      **kwargs):
    """Make a scorer that accepts span labels and predictions."""
    return make_scorer(make_modified_metric(metric, span_aligner, **kwargs))


def validate_scorers(scorers: t.List[str], span_aligner: SpanAligner):
    """Validate the given scorer list."""
    scoring_dict = get_scorer_dict(span_aligner)

    if not isinstance(scorers, Sequence):
        raise DeepchecksValueError(f'Scorers must be a Sequence, got {type(scorers)}')
    if not all(isinstance(name, str) for name in scorers):
        # TODO: support custom scorers
        raise DeepchecksValueError(f'Scorers must be a Sequence of strings, got {type(scorers[0])}')
    if any(name not in scoring_dict for name in scorers):
        raise DeepchecksValueError(f'Scorers must be a list of names of existing token classification metrics, which '
                                   f'is {scoring_dict.keys()}, got {scorers}')


def get_default_token_scorers(use_avg_defaults=True) -> t.List[str]:
    """Return the default scorers for token classification."""
    return DEFAULT_AVG_SCORER_NAMES if use_avg_defaults else DEFAULT_PER_CLASS_SCORER_NAMES