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
"""Module with common docstrings."""
from deepchecks.utils.decorators import Substitution

_shared_docstrings = {}


_shared_docstrings['additional_context_params'] = """
feature_importance: pd.Series , default: None
    pass manual features importance
feature_importance_force_permutation : bool , default: False
    force calculation of permutation features importance
feature_importance_timeout : int , default: 120
    timeout in second for the permutation features importance calculation
y_pred_train: Optional[np.ndarray] , default: None
    Array of the model prediction over the train dataset.
y_pred_test: Optional[np.ndarray] , default: None
    Array of the model prediction over the test dataset.
y_proba_train: Optional[np.ndarray] , default: None
    Array of the model prediction probabilities over the train dataset.
y_proba_test: Optional[np.ndarray] , default: None
    Array of the model prediction probabilities over the test dataset.
features_importance: Optional[pd.Series] , default: None
    pass manual features importance
    .. deprecated:: 0.8.1
        Use 'feature_importance' instead.
""".strip('\n')


docstrings = Substitution(**_shared_docstrings)
