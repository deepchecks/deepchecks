"""Module containing performance report check."""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlchecks import CheckResult, Dataset
from mlchecks.base.check import TrainValidationBaseCheck
from mlchecks.metric_utils import get_metrics_list, task_type_check, ModelType
from mlchecks.utils import model_type_validation

__all__ = ['naive_comparision', 'NaiveComparision']


def run_on_df(train_ds: Dataset, test_ds: Dataset, task_type: ModelType, model, NAIVE_MODEL_ORDER: int = 0, MAX_RATIO: float = 10):
        
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

        y_pred = model.predict(test_df.data[test_df.features()])
        y_test = test_df[label_col_name]

        if task_type == ModelType.REGRESSION:
            metric_type = 'mse'
            pred_metric = mean_squared_error(y_pred, y_test)
            naive_metric = mean_squared_error(naive_pred, y_test)

        elif task_type == ModelType.BINARY or task_type == ModelType.MULTICLASS:
            metric_type = 'log-loss'
            naive_metric = log_loss(y_test, naive_pred)
            pred_metric = log_loss(y_test, y_pred)
        else:
            raise (NotImplementedError(f'{task_type} not legal task_type'))

        res = min(pred_metric / naive_metric, MAX_RATIO) \
            if naive_metric != 0 else (1 if pred_metric == 0 else MAX_RATIO)

        model_type = 'regressor' if task_type == ModelType.REGRESSION else 'classifier'

        return res, metric_type, model_type

    
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

    # Get default metrics if no alternative, or validate alternatives
    metrics = get_metrics_list(model, dataset, alternative_metrics)
    scores = {key: scorer(model, dataset.features_columns(), dataset.label_col()) for key, scorer in metrics.items()}

    display_df = pd.DataFrame(scores.values(), columns=['Score'], index=scores.keys())
    display_df.index.name = 'Metric'

    return CheckResult(scores, check=self, display=display_df)


class NaiveComparision(TrainValidationBaseCheck):
    """Summarize given metrics on a dataset and model."""

    def run(self, train_dataset, validation_dataset, model) -> CheckResult:
        """Run performance_report check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format `{<metric>: score}`
        """
        return performance_report(train_dataset, validation_dataset, model)
