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
"""Utils module containing scorers for computing token classification metrics."""
import typing as t
from collections.abc import Sequence

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import Token
from sklearn.metrics import make_scorer

from deepchecks.core.errors import DeepchecksValueError

__all__ = ['get_default_token_scorers', 'validate_scorers', 'get_scorer_dict']

DEFAULT_AVG_SCORER_NAMES = ('f1_macro', 'recall_macro', 'precision_macro')
DEFAULT_PER_CLASS_SCORER_NAMES = ('f1_per_class', 'precision_per_class', 'recall_per_class')


def get_scorer_dict(
    suffix: bool = False,
    mode: t.Optional[str] = None,
    scheme: t.Optional[t.Type[Token]] = None,
) -> t.Dict[str, t.Callable[[t.List[str], t.List[str]], float]]:
    """Return a dict of scorers for token classification.

    Parameters
    ----------
    mode: str, [None (default), `strict`].
        if ``None``, the score is compatible with conlleval.pl. Otherwise,
        the score is calculated strictly.
    scheme: Token, [IOB2, IOE2, IOBES]
    suffix: bool, False by default.

    Returns
    -------
        A dict of scorers.
    """
    common_kwargs = {
        'mode': mode,
        'scheme': scheme,
        'suffix': suffix,
        'zero_division': 0,
    }

    return {
        'accuracy': make_token_scorer(accuracy_score, **common_kwargs),
        'f1_per_class': make_token_scorer(f1_score, **common_kwargs, average=None),
        'f1_macro': make_token_scorer(f1_score, **common_kwargs, average='macro'),
        'f1_micro': make_token_scorer(f1_score, **common_kwargs, average='micro'),
        'precision_per_class': make_token_scorer(precision_score, **common_kwargs, average=None),
        'precision_macro': make_token_scorer(precision_score, **common_kwargs, average='macro'),
        'precision_micro': make_token_scorer(precision_score, **common_kwargs, average='micro'),
        'recall_per_class': make_token_scorer(recall_score, **common_kwargs, average=None),
        'recall_macro': make_token_scorer(recall_score, **common_kwargs, average='macro'),
        'recall_micro': make_token_scorer(recall_score, **common_kwargs, average='micro'),
    }


def make_token_scorer(metric: t.Callable[[t.List[t.List[str]], t.List[t.List[str]]], float],
                      **kwargs):
    """Make a scorer that accepts span labels and predictions."""
    return make_scorer(metric, **kwargs)


def validate_scorers(scorers: t.List[str]):
    """Validate the given scorer list."""
    scoring_dict = get_scorer_dict()

    if not isinstance(scorers, Sequence):
        raise DeepchecksValueError(f'Scorers must be a Sequence, got {type(scorers)}')

    for name in scorers:
        if not isinstance(name, str):
            # TODO: support custom scorers
            raise DeepchecksValueError(
                f'Scorers must be a Sequence of strings, got {type(name)}'
            )
        if name not in scoring_dict:
            raise DeepchecksValueError(
                'Scorers must be a list of names of existing token classification metrics, '
                f'which is {scoring_dict.keys()}, got {scorers}'
            )


def get_default_token_scorers(use_avg_defaults=True) -> t.List[str]:
    """Return the default scorers for token classification."""
    return list(
        DEFAULT_AVG_SCORER_NAMES
        if use_avg_defaults
        else DEFAULT_PER_CLASS_SCORER_NAMES
    )
