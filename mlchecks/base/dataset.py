import pandas as pd
from pandas_profiling import ProfileReport

__all__ = ['Dataset']


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
