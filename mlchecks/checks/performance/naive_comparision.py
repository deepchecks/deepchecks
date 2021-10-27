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

def run_on_df(train_ds: Dataset, val_ds: Dataset, task_type: ModelType, model,
              native_model_type: str, metric = None, metric_name = None):
        """Find p value for column frequency change between the reference dataset to the test dataset.

        Args:
            train_ds (Dataset): The training dataset object. Must contain an index.
            val_ds (Dataset): The validation dataset object. Must contain an index.
            task_type (ModelType): the model type
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
            native_model_type (str):  Type of the naive model ['random' 'statistical' 'tree'].
        Returns:
            float: p value for the key.

        Raises:
            NotImplementedError: If the native_model_type is not a legal native_model_type

        """
        label_col_name = train_ds.label_name()
        features = train_ds.features()
        train_df = train_ds.data
        val_df = val_ds.data

        np.random.seed(0)

        if native_model_type == 'random':
            naive_pred = np.random.choice(train_df[label_col_name], val_df.shape[0])

        elif native_model_type == 'statistical':
            if task_type == ModelType.REGRESSION:
                naive_pred = np.array([np.mean(train_df[label_col_name])] * len(val_df))

            elif task_type == ModelType.BINARY or task_type == ModelType.MULTICLASS:
                counts = train_df[label_col_name].value_counts()
                naive_pred = np.array([counts.index[0]] * len(val_df))

        elif native_model_type == 'tree':
            X_train = train_df[features]
            y_train = train_df[label_col_name]
            X_test = val_df[features]

            if task_type == ModelType.REGRESSION:
                clf = DecisionTreeRegressor()
                clf = clf.fit(X_train, y_train)
                naive_pred = clf.predict(X_test)

            elif task_type == ModelType.BINARY or task_type == ModelType.MULTICLASS:

                clf = DecisionTreeClassifier()
                clf = clf.fit(X_train, y_train)
                naive_pred = clf.predict(X_test)
   
        else:
            raise (NotImplementedError(f'{native_model_type} not legal native_model_type'))

        y_test = val_df[label_col_name]
   
        if metric is not None:
            scorer = validate_scorer(metric, model, train_ds)
            metric_name = metric_name or metric if isinstance(metric, str) else 'User metric'
        else:
            metric_name = DEFAULT_SINGLE_METRIC[task_type]
            scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

        naive_metric = scorer(dummy_model, naive_pred, y_test)
        pred_metric = scorer(model, val_df[features], y_test)

        model_type = 'regressor' if task_type == ModelType.REGRESSION else 'classifier'

        return naive_metric, pred_metric, metric_name, model_type


def naive_comparision(train_dataset: Dataset, validation_dataset: Dataset,
                      model, native_model_type: str = 'random', max_ratio: float = 10):
    """Summarize given metrics on a dataset and model.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain an index.
        validation_dataset (Dataset): The validation dataset object. Must contain an index.
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        native_model_type (str = 'random'):  Type of the naive model ['random' 'statistical' 'tree'].
        max_ratio (float = 10):  Value to return in case the loss of the naive model is very low (or 0)
                                 and the loss of the predictions is positive (1 to inf).

    Returns:
        CheckResult: value is dictionary in format `{metric: score, ...}`
    
    Raises:
        MLChecksValueError: If the object is not a Dataset instance.
    """
    self = naive_comparision
    Dataset.validate_dataset(train_dataset, self.__name__)
    Dataset.validate_dataset(validation_dataset, self.__name__)
    train_dataset.validate_label(self.__name__)
    validation_dataset.validate_label(self.__name__)
    model_type_validation(model)

    naive_metric, pred_metric, metric_name, model_type = run_on_df(train_dataset, validation_dataset,
                                                                   task_type_check(model, train_dataset), model,
                                                                   native_model_type)

    res = min(pred_metric / naive_metric, max_ratio) \
            if naive_metric != 0 else (1 if pred_metric == 0 else max_ratio)

    text = f'Naive {model_type} has achieved {res:.2f} times ' \
           f'better {metric_name} compared to model prediction on tested data.'

    return CheckResult((naive_metric, pred_metric, metric_name, model_type), check=self, display=[text])


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
        return naive_comparision(train_dataset, validation_dataset, model, **self.params)
