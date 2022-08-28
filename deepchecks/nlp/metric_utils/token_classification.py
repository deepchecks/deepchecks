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

__all__ = ['get_default_token_scorers']


DEFAULT_AVG_SCORER_NAMES = ('f1_macro', 'recall_macro', 'precision_macro')
DEFAULT_PER_CLASS_SCORER_NAMES = ('f1_per_class', 'f1_per_class', 'f1_per_class')


def spans_to_token_list(y_true: TTokenLabel, y_pred: 'TTokenPred',
                        strict_alignment: bool = True, proba_threshold: float = 0
                        ) -> t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]:
    """Convert from predictions and labels in span format to a list of token annotations.

    Parameters:
    -----------
        y_true: TTokenLabel
            Labels as provided by the user in the accepted format
        y_pred: TTokenPred
            Predictions as provided by the user in the accepted format
        strict_alignment: bool, default: True
            If true, predictions are considered only if their spans match exactly any span in the labels. If false,
            predictions are considered if they overlap any label span.
        proba_threshold: float, default: 0
            The minimal probability score for a prediction to be considered.

    Returns:
    --------
        A dict of scorers.
    """
    if not isinstance(strict_alignment, bool):
        raise TypeError('strict_alignment must be a boolean')
    if not strict_alignment:
        # TODO: Implement non-strict alignment, probably using https://github.com/biocore-ntnu/ncls or similar
        raise NotImplementedError('Non Strict alignment is not yet implemented')

    if not len(y_true) == len(y_pred):
        raise DeepchecksValueError(f'Must have both a label entry and a prediction entry for each sample, but found '
                                   f'{len(y_true)} label entries and {len(y_pred)} prediction entries')

    labels = []
    predictions = []
    for sample_idx in range(len(y_true)):
        sample_y_true = sorted(y_true[sample_idx], key=lambda x: x[1])
        sample_y_pred = sorted(y_pred[sample_idx], key=lambda x: x[1])
        labels.append([annotation[0] for annotation in sample_y_true])

        label_spans = [(annotation[1], annotation[2]) for annotation in sample_y_true]
        pred_spans = [(annotation[1], annotation[2]) for annotation in sample_y_pred]

        if len(sample_y_true) > 0:
            # Indices of where to place elements of pred_spans in label_spans so that it's sorted
            sorted_index = np.searchsorted(list(map(lambda x: x[0], pred_spans)),
                                           list(map(lambda x: x[0], label_spans)), side='left')
            # Elements of sample_y_pred that match their counterparts in label_spans
            pred_index = np.take(np.arange(len(pred_spans)), sorted_index, mode='clip')
            bool_matching = np.array(list(map(hash, pred_spans)))[pred_index] == np.array(list(map(hash, label_spans)))
            # Filter out predictions that do not match their corresponding labels or their threshold is too low.
            # Predictions that where filtered are filled with 'O'.
            sample_predictions = \
                [sample_y_pred[pred_index[idx]][0] if
                 ((sample_y_pred[pred_index[idx]][3] >= proba_threshold) and bool_matching[idx]) else 'O'
                 for idx in range(len(sample_y_true))]
        else:
            sample_predictions = []

        predictions.append(sample_predictions)

    return labels, predictions


def make_modified_metric(metric: t.Callable[[t.List[t.List[str]], t.List[t.List[str]], t.Any], float],
                         strict_alignment: bool = True, proba_threshold: float = 0, **kwargs) -> t.Callable:
    """Apply spans_to_token_list processing to the labels and predictions, and then pass them to the metric function."""
    def modified_metric(y_true: TTokenLabel, y_pred: 'TTokenPred'):
        y_true_processed, y_pred_processed = spans_to_token_list(y_true, y_pred, strict_alignment, proba_threshold)
        return metric(y_true_processed, y_pred_processed, **kwargs)

    return modified_metric


def make_token_scorer(metric: t.Callable[[t.List[t.List[str]], t.List[t.List[str]]], float],
                      strict_alignment: bool = True, proba_threshold: float = 0,
                      **kwargs):
    """Make a scorer that accepts span labels and predictions."""
    return make_scorer(make_modified_metric(metric, strict_alignment, proba_threshold, **kwargs))


def get_scorer_dict(suffix: bool = False, mode: t.Optional[str] = None, scheme: t.Optional[t.Type[Token]] = None,
                    strict_alignment: bool = True, proba_threshold: float = 0
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
        'mode': mode,
        'scheme': scheme,
        'suffix': suffix,
        'zero_division': 0,
        'strict_alignment': strict_alignment,
        'proba_threshold': proba_threshold
    }

    return {
            'token_accuracy': make_token_scorer(accuracy_score),
            'token_f1_per_class':  make_token_scorer(f1_score, **common_kwargs, average=None),
            'token_f1_macro': make_token_scorer(f1_score, **common_kwargs, average='macro'),
            'token_f1_micro':  make_token_scorer(f1_score, **common_kwargs, average='micro'),
            'token_precision_per_class':  make_token_scorer(precision_score, **common_kwargs, average=None),
            'token_precision_macro': make_token_scorer(precision_score, **common_kwargs, average='macro'),
            'token_precision_micro':  make_token_scorer(precision_score, **common_kwargs, average='micro'),
            'token_recall_per_class':  make_token_scorer(recall_score, **common_kwargs, average=None),
            'token_recall_macro': make_token_scorer(recall_score, **common_kwargs, average='macro'),
            'token_recall_micro':  make_token_scorer(recall_score, **common_kwargs, average='micro'),
           }


def get_default_token_scorers(scorers: t.List[str], use_avg_defaults=True
                              ) -> t.Dict[str, t.Callable[[t.List[str], t.List[str]], float]]:
    """Return the default scorers for token classification."""
    scoring_dict = get_scorer_dict()
    names_to_get = scorers or (DEFAULT_AVG_SCORER_NAMES if use_avg_defaults else DEFAULT_PER_CLASS_SCORER_NAMES)
    if not isinstance(names_to_get, Sequence):
        raise DeepchecksValueError(f'Scorers must be a Sequence, got {type(scorers)}')
    if not all(isinstance(name, str) for name in names_to_get):
        raise DeepchecksValueError(f'Scorers must be a Sequence of strings, got {type(names_to_get[0])}')
    if any(name not in scoring_dict for name in names_to_get):
        raise DeepchecksValueError(f'Scorers must be a list of names of existing token classification metrics, which '
                                   f'is {scoring_dict.keys()}, got {names_to_get}')
    return {name: scoring_dict[name] for name in names_to_get}
