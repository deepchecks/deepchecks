## Example Check Notebooks

Using the checks, you can examine different aspects of your data and models. Here is a list with some examples of checks and a link to the notebooks demonstrating them.
  

 ## Checks List

  Note that this table is currently partial, however all checks can be found in the example notebooks in this folder.

Category | Check | Description |
|---------|---------|----------------|
| Performance | [CalibrationMetric](./performance_examples/calibration_metric.ipynb)|Summarize given metrics on a dataset and model |
| Performance | [Performance Report](./performance_examples/performance_report_example.ipynb)|Summarize given metrics on a dataset and model |
| Performance | [ROC Report](./performance_examples/roc_report_example.ipynb)|Return the AUC for each class ||
| Performance| [ConfusionMatrixReport](./performance_examples/confusion_matrix_report_example.ipynb)|Return the confusion_matrix |
| Overfit| [PerformanceOverfit](./performance_overfit.ipynb) | Visualize overfit by displaying the difference between model metrics on train and on validation data |
| Overfit| [Boosting overfit](./boosting_overfit.ipynb) |Check for overfit occurring when increasing the number of iterations in boosting models
| Leakage| [IndexLeakageReport](./Index%20Leakage.ipynb)|Check if validation indexes are present in train data |
| Leakage | [SingleFeatureContribution](./single_feature_contribution.ipynb) | Return the PPS (Predictive Power Score) of all features in relation to the label |
| Leakage| [DataSampleLeakage](./data_sample_leakage.ipynb) |Find what percent of the validation data is in the train data |
| Integrity| [DominantFrequencyChange](./dominant_frequency_change.ipynb) |Find what percent of the validation data is in the train data |
| Integrity| [MixedNulls](./mixed_nulls.ipynb) | Search for various types of null values in a string column(s), including string representations of null |
| Integrity| [MixedTypes](./mixed_types.ipynb) | Search for mixed types of Data in a single column |
| Integrity| [NewCategory](./new_category.ipynb) | Find new categories in validation |
| Integrity| [RareFormatDetection](./rare_format_detection.ipynb) | Check whether columns have common formats (e.g. "XX-XX-XXXX" for dates") and detects values that don't match |
| Integrity| [SpecialCharacters](./special_characters.ipynb) | Search in column[s] for values that contains only special characters |
| Integrity| [StringMismatch](./String%20mismatch.ipynb)| Detect different variants of string categories (e.g. "mislabeled" vs "mis-labeled") in a categorical column |
| Integrity|[StringMismatchComparison](./string_mismatch_comparison.ipynb) | Detect different variants of string categories between the same categorical column in two datasets|
| Integrity|[DataDuplicates](./data_duplicats.ipynb) |Search for Data duplicates in dataset |