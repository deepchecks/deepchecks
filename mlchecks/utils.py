import sklearn
import catboost


class MLChecksException(Exception):
    pass


SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]