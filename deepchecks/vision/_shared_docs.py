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
train_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
test_predictions: Optional[Dict[int, Union[Sequence[torch.Tensor], torch.Tensor]]] , default None
""".strip('\n')


docstrings = Substitution(**_shared_docs)
