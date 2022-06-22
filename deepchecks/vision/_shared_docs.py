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

_shared_docs = {}


_shared_docs['additional_context_params'] = """
model_name: str , default: ''
    The name of the model
scorers : Optional[Mapping[str, Metric]] , default: None
    dict of scorers names to a Metric
scorers_per_class : Optional[Mapping[str, Metric]] , default: None
    dict of scorers for classification without averaging of the classes.
    See <a href=
    "https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel">
    scikit-learn docs</a>
device : Union[str, torch.device], default: 'cpu'
    processing unit for use
random_state : int
    A seed to set for pseudo-random functions
n_samples : Optional[int], default: None
    number of samples
with_display : bool , default: True
    flag that determines if checks will calculate display (redundant in some checks).
train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
    Dictionary of the model prediction over the train dataset (keys are the indexes).
test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
    Dictionary of the model prediction over the test dataset (keys are the indexes).
""".strip('\n')


docstrings = Substitution(**_shared_docs)
