"""Module containing naive comparision check."""
import numpy as np
import matplotlib.pyplot as plt

from mlchecks import CheckResult, Dataset
from mlchecks.base.check import TrainValidationBaseCheck
from mlchecks.metric_utils import DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, ModelType, validate_scorer
from mlchecks.utils import model_type_validation

__all__ = ['NaiveModelComparison']


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
                                random is random label from train,
                                statistical is mean for regression or
                                the label that appears most often for classification
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
            counts = train_ds.label_col().mode()
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


class NaiveModelComparison(TrainValidationBaseCheck):
    """Compare naive model score to given model score."""

    def __init__(self, naive_model_type: str = 'statistical', metric=None, metric_name=None):
        """Initialize the NaiveModelComparison check.

        Args:
            naive_model_type (str = 'random'):  Type of the naive model ['random', 'statistical'].
            metric: a custom metric given by user.
            metric_name: name of a default metric.
        """
        super().__init__()
        self.naive_model_type = naive_model_type
        self.metric = metric
        self.metric_name = metric_name

    def run(self, train_dataset, validation_dataset, model) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            validation_dataset (Dataset): The validation dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.

        Returns:
            CheckResult: value is ratio between model prediction to naive prediction

        Raises:
            MLChecksValueError: If the object is not a Dataset instance.
        """
        return self._naive_model_comparison(train_dataset, validation_dataset, model)

    def _naive_model_comparison(self, train_dataset: Dataset, validation_dataset: Dataset, model):
        func_name = self.__class__.__name__
        Dataset.validate_dataset(train_dataset, func_name)
        Dataset.validate_dataset(validation_dataset, func_name)
        train_dataset.validate_label(func_name)
        validation_dataset.validate_label(func_name)
        model_type_validation(model)

        naive_metric, pred_metric, metric_name = find_score(train_dataset, validation_dataset,
                                                            task_type_check(model, train_dataset), model,
                                                            self.naive_model_type, self.metric,
                                                            self.metric_name)

        text = f'{type(model).__name__} Model prediction has achieved {pred_metric} ' \
               f'in {metric_name} compared to Naive {self.naive_model_type} prediction ' \
               f'which achived {naive_metric} on tested data.'

        def display_func():
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            models = [f'Naive model - {self.naive_model_type}', f'{type(model).__name__} model']
            metrics_results = [naive_metric, pred_metric]
            ax.bar(models, metrics_results)
            ax.set_ylabel(metric_name)

        return CheckResult({'given_model_score': pred_metric, 'naive_model_score': naive_metric},
                           check=self.__class__, display=[text, display_func])
