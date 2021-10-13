from mlchecks import Dataset
import pandas as pd

from mlchecks.base.check import CheckResult, TrainValidationBaseCheck
from mlchecks.display import format_check_display


def get_dup_indexes_map(df, features):
    dup = df[df.duplicated(features, keep=False)].groupby(features).groups.values()
    dup_map = dict()
    for i_arr in dup:
        key = i_arr[0]
        dup_map[key] = [int(i) for i in i_arr[1:]]
    return dup_map


def get_dup_txt(i, dup_map):
    val = dup_map.get(i)
    if not val:
        return i
    txt = f'{i}, '
    for j in val:
        txt += f'{j}, '
    return txt[:-2]


def data_leakage_report(validation_dataset: Dataset, test_dataset: Dataset):
    features = test_dataset.features()
    test_f = test_dataset[features]
    val_f = validation_dataset[features]

    test_dups = get_dup_indexes_map(test_dataset, features)
    test_f.index = [f'test indexs: {get_dup_txt(i, test_dups)}' for i in test_f.index]
    test_f.drop_duplicates(features, inplace = True)

    val_dups = get_dup_indexes_map(val_f, features)
    val_f.index = [f'validation indexs: {get_dup_txt(i, val_dups)}' for i in val_f.index]
    val_f.drop_duplicates(features, inplace = True)

    appended_df = val_f.append(test_f)
    duplicateRowsDF = appended_df[appended_df.duplicated(features, keep=False)]
    duplicateRowsDF.sort_values(features, inplace=True)
    
    return CheckResult(duplicateRowsDF.to_dict(), display={'text/html': format_check_display('Classification Report', data_leakage_report, duplicateRowsDF.to_html())})

class ClassificationReport(TrainValidationBaseCheck):
    """Summarize given model parameters."""

    def run(self, dataset: Dataset) -> CheckResult:
        """Run classification_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult: value is dictionary in format {<target>: , ['precision': <score>, 'recall': <score>, 'f_score': <score>, 'support': <score>]}
        """
        return data_leakage_report(dataset)
