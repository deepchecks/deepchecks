"""Module containing naive comparision check."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt

from mlchecks import CheckResult, Dataset
from mlchecks.base.check import TrainValidationBaseCheck
from mlchecks.metric_utils import DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, ModelType, validate_scorer
from mlchecks.utils import model_type_validation

__all__ = ['naive_comparison', 'NaiveComparison']


class DummyModel():
    @staticmethod
    def predict(a):
        return a
    @staticmethod
    def predict_proba(a):
        return a

def find_score(train_ds: Dataset, val_ds: Dataset, task_type: ModelType, model,
              naive_model_type: str, metric = None, metric_name = None):
    """Find the naive model score for given metric.

    Args:
        train_ds (Dataset): The training dataset object. Must contain an index.
        val_ds (Dataset): The validation dataset object. Must contain an index.
        task_type (ModelType): the model type.
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        naive_model_type (str): Type of the naive model ['random', 'statistical'].
        metric: a custom metric given by user.
        metric_name: name of a default metric.
    Returns:
        float: p value for the key.

    Raises:
        NotImplementedError: If the naive_model_type is not supported

    """
    val_df = val_ds.data

    np.random.seed(0)

    if naive_model_type == 'random':
        naive_pred = np.random.choice(train_ds.label_col(), val_df.shape[0])

    elif naive_model_type == 'statistical':
        if task_type == ModelType.REGRESSION:
            naive_pred = np.array([np.mean(train_ds.label_col())] * len(val_df))

        elif task_type in (ModelType.BINARY, ModelType.MULTICLASS):
            counts = train_ds.label_col().value_counts()
            naive_pred = np.array([counts.index[0]] * len(val_df))

    else:
        raise NotImplementedError(f"expected to be one of ['random', 'statistical'] \
                                   but instaed got {naive_model_type}")

    y_val = val_ds.label_col()

    if metric is not None:
        scorer = validate_scorer(metric, model, train_ds)
        metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
    else:
        metric_name = DEFAULT_SINGLE_METRIC[task_type]
        scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

    naive_metric = scorer(DummyModel, naive_pred, y_val)
    pred_metric = scorer(model, val_ds.features_columns(), y_val)

    return naive_metric, pred_metric, metric_name


def naive_comparison(train_dataset: Dataset, validation_dataset: Dataset,
                      model, naive_model_type: str = 'random', max_ratio: float = 10,
                      metric = None, metric_name = None):
    """Compare naive model score to given model score.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain a label.
        validation_dataset (Dataset): The validation dataset object. Must contain a label.
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        naive_model_type (str = 'random'):  Type of the naive model ['random', 'statistical'].
        max_ratio (float = 10):  Value to return in case the score of the naive model is very low (or 0)
                                 and the score of the predictions is positive (1 to inf).
        metric: a custume metric given by user.
        metric_name: name of a default metric.

    Returns:
        CheckResult: value is ratio between model prediction to naive prediction

    Raises:
        MLChecksValueError: If the object is not a Dataset instance.
    """
    self = naive_comparison
    Dataset.validate_dataset(train_dataset, self.__name__)
    Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    validation_dataset.validate_label(self.__name__)
    model_type_validation(model)

    naive_metric, pred_metric, metric_name = find_score(train_dataset, validation_dataset,
                                                       task_type_check(model, train_dataset), model,
                                                       naive_model_type, metric, metric_name)

    res = min(pred_metric / naive_metric, max_ratio) \
            if naive_metric != 0 else (1 if pred_metric == 0 else max_ratio)

    text = f'Model prediction has achieved {res:.2f} times ' \
           f'more {metric_name} compared to Naive {naive_model_type} prediction on tested data.'

    def display_func():
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        models = [f'Naive model - {naive_model_type}', f'{type(model).__name__} model']
        metrics_results = [naive_metric, pred_metric]
        ax.bar(models,metrics_results)
        ax.set_ylabel(metric_name)

    return CheckResult(res, check=self, display=[text, display_func])


class NaiveComparison(TrainValidationBaseCheck):
    """Compare naive model score to given model score."""

    def run(self, train_dataset, validation_dataset, model) -> CheckResult:
        """Run naive_comparision check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            validation_dataset (Dataset): The validation dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.

        Returns:
            CheckResult: value is ratio between model prediction to naive prediction
        """
        return naive_comparison(train_dataset, validation_dataset, model, **self.params)
