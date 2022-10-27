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
model_classes: Optional[List] , default: None
    For classification: list of classes known to the model
""".strip('\n')

_shared_docstrings['feature_aggregation_method_argument'] = """
argument for the reduce_output functionality, decides how to aggregate the drift scores for a
collective score. The collective score value is between 0 and 1 for all methods other than l2_combination.
Possible values are:
'l2_weighted': L2 norm over the combination of drift scores and feature importance, minus the
L2 norm of feature importance alone, specifically, ||FI + DRIFT|| - ||FI||. This method returns a
value between 0 and sqrt(n_features).
'weighted': Weighted mean based on feature importance, provides a robust estimation on how
much the drift will affect the model's performance.
'mean': Mean of all drift scores.
'max': Maximum of all the features drift scores.
'none': No averaging. Return a dict with a drift score for each feature.
'top_5' No averaging. Return a dict with a drift score for top 5 features based on feature importance.
""".strip('\n')

docstrings = Substitution(**_shared_docstrings)
