from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck


def dataset_info(dataset: Dataset):
    """Summarize given dataset information based on pandas_profiling package
       Args:
           dataset (Dataset): A dataset object
       Returns:
           CheckResult: value is tuple that represents the shape of the dataset
    """
    return CheckResult(dataset.shape, display={'text/html': dataset.get_profile().to_notebook_iframe()})


class DatasetInfo(SingleDatasetBaseCheck):
    """
    Summarize given dataset information based on pandas_profiling package
    Can be used inside `Suite`
    """
    def run(self, dataset, model=None) -> CheckResult:
        return dataset_info(dataset)

