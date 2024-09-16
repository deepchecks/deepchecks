# -*- coding: utf-8 -*-
"""
.. _nlp__confusion_matrix_report:

Confusion Matrix Report
***********************
This notebook provides an overview for using and understanding the Confusion Matrix Report check for NLP tasks.


**Structure:**

* `What is the Confusion Matrix Report? <#what-is-the-confusion-matrix-report>`__
* `Generate data & model <#generate-data-model>`__
* `Run the check <#run-the-check>`__


What is the Confusion Matrix Report?
======================================
The ``ConfusionMatrixReport`` produces a confusion matrix visualization which summarizes the
performance of the model. The confusion matrix contains the TP (true positive), FP (false positive),
TN (true negative) and FN (false negative), from which we can derive the relevant metrics,
such as accuracy, precision, recall etc. (`confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`__).
"""

# %%
# Generate data & model
# =======================
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import ConfusionMatrixReport
from deepchecks.nlp.datasets.classification.tweet_emotion import load_data, load_precalculated_predictions

tweets_data = load_data(data_format="DataFrame", as_train_test=False)
tweets_dataset = TextData(tweets_data.text, label=tweets_data["label"], task_type="text_classification")

predictions = load_precalculated_predictions(as_train_test=False)


# %%
# Run the check
# ===============

check = ConfusionMatrixReport()
result = check.run(tweets_dataset, predictions=predictions)
result.show()

# %%
# Define a condition
# ==================
# We can define our check a condition that will validate if all the misclassified
# cells/samples in the confusion matrix is below a certain threshold. Using the
# ``misclassified_samples_threshold`` argument, we can specify what percentage of the total samples
# our condition should use to validate the misclassified cells.

# Let's add a condition and re-run the check:

check = ConfusionMatrixReport()
check.add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=0.1)
result = check.run(tweets_dataset, predictions=predictions)
result.show()

# %%
