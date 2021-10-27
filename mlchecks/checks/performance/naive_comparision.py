"""Module containing performance report check."""
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array

from mlchecks import CheckResult, Dataset
from mlchecks.base.check import TrainValidationBaseCheck
from mlchecks.metric_utils import DEFAULT_BINARY_METRICS, DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, ModelType, validate_scorer
from mlchecks.utils import model_type_validation

__all__ = ['naive_comparision', 'NaiveComparision']


class dummy_model():
    def predict(a):
        return a
    def predict_proba(a):
        return a

def run_on_df(train_ds: Dataset, test_ds: Dataset, task_type: ModelType, model,
              metric = None, metric_name = None,
              NAIVE_MODEL_ORDER: int = 0, MAX_RATIO: float = 10):
        
        label_col_name = train_ds.label_name()
        features = train_ds.features()
        train_df = train_ds.data
        test_df = test_ds.data

        np.random.seed(0)

        if NAIVE_MODEL_ORDER == 0:
            naive_pred = np.random.choice(train_df[label_col_name], test_df.shape[0])

        elif NAIVE_MODEL_ORDER == 1:
            if task_type == ModelType.REGRESSION:
                naive_pred = np.array([np.mean(train_df[label_col_name])] * len(test_df))

            elif task_type == ModelType.BINARY or task_type == ModelType.MULTICLASS:

                counts = train_df[label_col_name].value_counts()
                naive_pred = np.array([counts.index[0]] * len(test_df))

        elif NAIVE_MODEL_ORDER == 2:
            X_train = train_df[features]
            y_train = train_df[label_col_name]
            X_test = test_df[features]

            if task_type == ModelType.REGRESSION:
                clf = DecisionTreeRegressor()
                clf = clf.fit(X_train, y_train)
                naive_pred = clf.predict(X_test)

            elif task_type == ModelType.BINARY or task_type == ModelType.MULTICLASS:

                clf = DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)
                naive_pred = clf.predict(X_test)
        
        else:
            raise (NotImplementedError(f'{NAIVE_MODEL_ORDER} not legal NAIVE_MODEL_ORDER'))

        y_test = test_df[label_col_name]
        
        if metric is not None:
            scorer = validate_scorer(metric, model, train_ds)
            metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
        else:
            metric_name = DEFAULT_SINGLE_METRIC[task_type]
            scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

        naive_metric = scorer(dummy_model, naive_pred, y_test)
        pred_metric = scorer(model, test_df[features], y_test)

        res = min(pred_metric / naive_metric, MAX_RATIO) \
            if naive_metric != 0 else (1 if pred_metric == 0 else MAX_RATIO)

        model_type = 'regressor' if task_type == ModelType.REGRESSION else 'classifier'

        return res, metric_name, model_type


def naive_comparision(train_dataset: Dataset, validation_dataset: Dataset, model):
    """Summarize given metrics on a dataset and model.

    Args:
        dataset (Dataset): a Dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions. If none
        given, using default metrics

    Returns:
        CheckResult: value is dictionary in format `{metric: score, ...}`
    """
    self = naive_comparision
    Dataset.validate_dataset(train_dataset, self.__name__)
    Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    validation_dataset.validate_label(self.__name__)
    model_type_validation(model)

    value = run_on_df(train_dataset, validation_dataset, task_type_check(model, train_dataset), model)

    return CheckResult(value, check=self, display=None)


class NaiveComparision(TrainValidationBaseCheck):
    """Summarize given metrics on a dataset and model."""

    def run(self, train_dataset, validation_dataset, model) -> CheckResult:
        """Run naive_comparision check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format `{<metric>: score}`
        """
        return naive_comparision(train_dataset, validation_dataset, model)
