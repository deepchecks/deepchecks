"""
=========================================================
Model Evaluation Suite on Breast Cancer Data - Quickstart
=========================================================
The model evaluation suite is relevant when you have a new model that you'd like to analyze for when it works, when it
doesn't, and how you might improve it. Here we’ll use the avocado prices dataset, to demonstrate how you can run the
suite with only a few simple lines of code, and see which kind of insights it can find.

"""
from deepchecks.tabular import Dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from deepchecks.tabular.checks.performance import ConfusionMatrixReport

from deepchecks.tabular import datasets

# %%
# Loading Data
# ============
data = datasets.classification.breast_cancer.load_data(data_format='Dataframe', as_train_test=False)

# %%
# Let's create a scikit-learn AdaBoost model with all the standard settings, then run it on our breast cancer data
# for evaluation:

clf = AdaBoostClassifier()
X = data[['mean radius', 'mean texture', 'mean perimeter', 'mean area',
          'mean smoothness', 'mean compactness', 'mean concavity',
          'mean concave points', 'mean symmetry', 'mean fractal dimension',
          'radius error', 'texture error', 'perimeter error', 'area error',
          'smoothness error', 'compactness error', 'concavity error',
          'concave points error', 'symmetry error', 'fractal dimension error',
          'worst radius', 'worst texture', 'worst perimeter', 'worst area',
          'worst smoothness', 'worst compactness', 'worst concavity',
          'worst concave points', 'worst symmetry', 'worst fractal dimension']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf.fit(X_train, y_train)

ds_train = Dataset(X_train, label=y_train, cat_features=[])
ds_test = Dataset(X_test, label=y_test, cat_features=[])

# %%
# Run the Deepchecks Suite
# =========================
#
# Validate your data with the deepchecks.tabular.suites.model_evaluation suite. It requires your training and test
# datasets as well as the model you used!
#
# Check out the
# [when should you use deepchecks guide](https://docs.deepchecks.com/dev/getting-started/when_should_you_use.html) for
# some more info about the existing suites and when to use them.

from deepchecks.tabular.suites import model_evaluation

eval_suite = model_evaluation()
eval_suite.run(ds_train, ds_test, clf)

# %%
# Model Evaluation Suite
# ======================
#
# The suite is composed of various checks such as: Unused Features, Performance Report, ROC Report, etc...
#
#
# Each check may contain conditions (which will result in pass / fail / warning / error , represented by ✓ / ✖ / ! / ⁈ )
# as well as other outputs such as plots or tables. Suites, checks and conditions can all be modified. Read more about
# [custom suites](https://docs.deepchecks.com/en/stable/user-guide/general/customizations/examples/plot_create_a_custom_suite.html).
#
# What does this suite find?
# ---------------------------
#
# We can inspect the suite outputs and see where our model could be improved.
#
# Let's start with the unused features.
# -------------------------------------

from deepchecks.tabular.checks import UnusedFeatures

# first let's see how the check runs:
UnusedFeatures().run(ds_train, ds_test, clf)

# %%
# We see here that some features like 'smoothness error', 'concavity error', and 'texture error' have lower feature
# importance but higher variance than some of the other features in the model.
#
# Why is this? For some of these features, the result makes sense: for example the ones with 'error' on the end measure
# the _uncertainty_ associated with measures quantifying characteristics of the breast tissue, as opposed to the
# characteristics themselves. We would not expect the certainty of our metrics to impact the presence of breast cancer.
#
# Others, like 'worst compactness,' may be worth looking into. We would want to understand what that feature means to
# get a better picture of whether this feature's low feature importance makes sense. Perhaps if this feature _should_be
# predictive, for example, our data on it contains enough noise as to carry very little real information about it. If
# that's the case we may want to explore a better way to collect this metric, for example.
#
# For now let's remove those error columns.


X = X.drop(['smoothness error', 'concavity error', 'texture error'], axis=1)

# %%
# Let's next look at a check that did not have output: Model Error Analysis.
# --------------------------------------------------------------------------
#
# That check appears in the "Checks without Output" section with the message 'Unable to train meaningful error model
# (r^2 score: 0.48)'. What does it mean?

from deepchecks.tabular.checks import ModelErrorAnalysis

# first let's see how the check runs:
ModelErrorAnalysis().run(ds_train, ds_test, clf)

# %%
# This is saying that only 48% of the variation in the target variable can be explained by the features that the model
# has access to.
#
# For data with clear, consistent identification mechanisms—like, say, the iris dataset—this would be weird. We expect
# the iris plant's physical characteristics to pretty clearly map to the variety of plant it is.
#
# Cancer is notoriously much, much more complicated. Though we know some risk factors, we cannot always predict when it
# will occur or how it will present.
#
# Though this does not necessarily mean something is missing from our data, it _does_ mean that this check cannot do
# what it helps to do: identify whether there are certain features that can predict whether the _existing_ model is
# going to get its classification _wrong_. Suppose, for example, that the dataset included a boolean feature like
# `dense_breast_tissue`. Because many metrics are harder to collect accurately for dense breast tissue, it's possible
# that the metrics we _do_ collect would be less representative of the presence of cancer in those cases. This check
# would flag something like that and clue us into a need for better, or maybe different, metric collection mechanisms
# for dense breast tissue.
#
# Finally, let's take a closer look at the Performance Report Check.
# ------------------------------------------------------------------
#
# Our data, unadulterated, passed this check. But what would make it fail? Suppose our data had a column indicating the
# presence of paclitaxel in the patient's blood sample:


X_train['paclitaxel_in_blood_sample'] = y_train
X_test['paclitaxel_in_blood_sample'] = 0

# %%
# What I've done above is a drastic oversimplification that assumes the use of paclitaxel, a common chemotherapy drug,
# for _all_ patients in whom breast cancer is detected.
#
# But it's classic temporal leakage: that treatment would have happened _after_ and _because_ breast cancer was
# detected, making it a feature that should _not_ be included in a model like this even if it's in a mass dump of
# patient data. We'd have it for the training data, but not for new data where we'd be asking the model for a prediction
# because those patients would _not_ have been diagnosed or treated yet.
#
# What happens when we run our suite with this new, leaky feature in there?

clf.fit(X_train, y_train)

ds_train = Dataset(X_train, label=y_train, cat_features=[])
ds_test = Dataset(X_test, label=y_test, cat_features=[])

eval_suite = model_evaluation()
eval_suite.run(ds_train, ds_test, clf)

# %%
# _Now_ we see a failure because the model performs _so_ much more poorly on the test data due to the fact that the
# paclitaxel feature is predictive for training data, but not for new data without the leakage.
#
# Let's restore our data and re-run our suite:

X_train = X_train.drop(['paclitaxel_in_blood_sample', 'smoothness error', 'concavity error', 'texture error'], axis=1)
X_test = X_test.drop(['paclitaxel_in_blood_sample', 'smoothness error', 'concavity error', 'texture error'], axis=1)

clf.fit(X_train, y_train)

ds_train = Dataset(X_train, label=y_train, cat_features=[])
ds_test = Dataset(X_test, label=y_test, cat_features=[])

eval_suite = model_evaluation()
eval_suite.run(ds_train, ds_test, clf)
