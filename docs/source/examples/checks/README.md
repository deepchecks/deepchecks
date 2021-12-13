## Example Check Notebooks

Using the checks, you can examine different aspects of your data and models. Here is a list with some examples of checks and a link to the notebooks demonstrating them.
  

 ## Checks List

  Note that this table is currently partial, however all checks can be found in the example notebooks in this folder.

Category | Check | Description |
|---------|---------|----------------|
| Performance | [SimpleModelComparison](./performance/simple_model_comparison.ipynb)|Compare simple model score to given model score |
| Performance | [CalibrationMetric](./performance/calibration_metric.ipynb)|Summarize given metrics on a dataset and model |
| Performance | [Performance Report](./performance/performance_report.ipynb)|Summarize given metrics on a dataset and model |
| Performance | [ROC Report](./performance/roc_report.ipynb)|Return the AUC for each class |
| Performance| [ConfusionMatrixReport](./performance/confusion_matrix_report.ipynb)|Return the confusion_matrix |
| Methodology| [PerformanceOverfit](./methodology/performance_overfit.ipynb) | Visualize overfit by displaying the difference between model metrics on train and on validation data |
| Methodology| [BoostingOverfit](./methodology/boosting_overfit.ipynb) |Check for overfit occurring when increasing the number of iterations in boosting models
| Methodology| [UnusedFeatures](./methodology/unused_features.ipynb) |Detect features that are nearly unused by the model |
| Methodology | [IndexLeakageReport](./methodology/index_leakage.ipynb)|Check if validation indexes are present in train data |
| Methodology | [SingleFeatureContribution](./methodology/single_feature_contribution.ipynb) | Return the PPS (Predictive Power Score) of all features in relation to the label |
| Methodology| [DataSampleLeakage](./methodology/data_sample_leakage.ipynb) |Find what percent of the validation data is in the train data |
| Integrity| [DominantFrequencyChange](./integrity/dominant_frequency_change.ipynb) |Find what percent of the validation data is in the train data |
| Integrity| [MixedNulls](./integrity/mixed_nulls.ipynb) | Search for various types of null values in a string column(s), including string representations of null |
| Integrity| [MixedTypes](./integrity/mixed_types.ipynb) | Search for mixed types of Data in a single column |
| Integrity| [NewCategory](./integrity/new_category.ipynb) | Find new categories in validation |
| Integrity| [RareFormatDetection](./integrity/rare_format_detection.ipynb) | Check whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match |
| Integrity| [SpecialCharacters](./integrity/special_characters.ipynb) | Search in column[s] for values that contains only special characters |
| Integrity| [StringMismatch](./integrity/string_mismatch.ipynb)| Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column |
| Integrity|[StringMismatchComparison](./integrity/string_mismatch_comparison.ipynb) | Detect different variants of string categories between the same categorical column in two datasets|
| Integrity|[DataDuplicates](./integrity/data_duplicates.ipynb) |Search for Data duplicates in dataset |