import typing as t
import pandas as pd

from deepchecks import TrainTestBaseCheck, Dataset, CheckResult
from deepchecks.metric_utils import task_type_validation, ModelType


MetricFunc = t.Callable[
    [object, pd.DataFrame, pd.Series], # model, features, labels
    float
]


class ClassPerformanceImbalanceCheck(TrainTestBaseCheck):

    def __init__(self, metrics: t.Optional[MetricFunc] = None):
        self.metric = metrics

    def run(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        model: object # TODO: find more precise type for model
    ) -> CheckResult:
        check_name = type(self).__name__
        Dataset.validate_dataset(train_dataset, check_name)
        Dataset.validate_dataset(test_dataset, check_name)
        task_type_validation(
            model=model, 
            dataset=train_dataset,
            expected_types=[ModelType.BINARY, ModelType.MULTICLASS]
        )

