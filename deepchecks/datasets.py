import warnings

from deepchecks.tabular.datasets import classification, regression

__all__ = [
    'classification',
    'regression',
    ]

warnings.warn(
    # TODO: better message
    'Ability to import base tabular functionality from '
    'the `deepchecks` directly is deprecated, please import from '
    '`deepchecks.tabular` instead',
    DeprecationWarning
)