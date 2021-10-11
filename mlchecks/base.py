import abc
from typing import Callable, Dict, List, Any, Tuple
import pandas as pd
from pandas_profiling import ProfileReport

__all__ = ['CheckResult', 'Checkable', 'Check', 'CheckSuite', 'Dataset']


class CheckResult:
    value: Any
    display: Dict

    def __init__(self, value, display=None):
        """
        Args:
            value (Any):
            display (Dict): Dictionary with formatters for display. possible
            foramtters are: 'text/html', 'image/png'
        """
        self.value = value
        self.display = display

    def _repr_mimebundle_(self, include, exclude):
        if not self.display:
            return None
        data = self.display
        if include:
            data = {k:v for (k,v) in data.items() if k in include}
        if exclude:
            data = {k:v for (k,v) in data.items() if k not in exclude}
        return data

    def __repr__(self):
        return self.value


class Checkable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, model=None, train_data=None, validation_data=None) -> CheckResult:
        pass


class Decidable(Checkable):
    """
    Check is a utility class which gives the option to combine a check and a decision function to be used together
    """
    deciders: List[Callable]

    def __init__(self, deciders: List[Callable]=None, **check_params):
        self.deciders = deciders or []
        self.check_params = check_params

    def run_and_decide(self, model=None, train_data=None, validation_data=None) -> Tuple[CheckResult, bool]:
        result = self.run(model=model, train_data=train_data, validation_data=validation_data)
        decisions = [x(result.value) for x in self.deciders] or None
        return result, decisions

    def decider(self, decider: Callable[[Any], bool]):
        deciders = [*self.deciders, decider]
        return self.__class__(deciders=deciders, **self.check_params)



class CheckSuite:
    checks: List[Checkable]

    def __init__(self, *checks):
        for check in checks:
            if not isinstance(check, Checkable):
                raise Exception(f'CheckSuite receives only `Checkable` objects but got: {check.__class__.__name__}')
        self.checks = checks

    def run(self, model=None, train_data=None, validation_data=None):
        return [check.run(model=model, train_data=train_data, validation_data=validation_data)
                for check in self.checks]

    def run_and_decide(self, model=None, train_data=None, validation_data=None):
        return [check.run_and_decide(model=model, train_data=train_data, validation_data=validation_data)
                for check in self.checks]


class Dataset(pd.DataFrame):
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str] = None, cat_features: List[str] = None,
                 label: str = None, index: str = None, date: str = None,
                 *args, **kwargs):

        super().__init__(df, *args, **kwargs)

        if features:
            self._features = features
        else:
            self._features = [x for x in df.columns if x not in {label, index, date}]

        self._label = label
        self._index_name = index
        self._date_name = date

        if cat_features:
            self._cat_features = cat_features
        else:
            self._cat_features = self.infer_categorical_features()

    def infer_categorical_features(self) -> List[str]:
        # TODO: add infer logic here
        return []

    def features(self) -> List[str]:
        return self._features

    def index_name(self) -> str:
        return self._index_name

    def date_name(self) -> str:
        return self._date_name

    def cat_features(self) -> List[str]:
        return self._cat_features

    def _get_profile(self):
        profile = ProfileReport(self, title="Dataset Report", explorative=True, minimal=True)
        return profile

    def _repr_mimebundle_(self, include, exclude):
        return {'text/html': self._get_profile().to_notebook_iframe()}
