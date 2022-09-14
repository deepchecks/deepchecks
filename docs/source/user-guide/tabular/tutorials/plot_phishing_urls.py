# -*- coding: utf-8 -*-
"""
Use Cases - Classifying Malicious URLs
**************************************

This notebook demonstrates how the ``deepchecks`` package can help you validate
your basic data science workflow right out of the box!

The scenario is a real business use case: You work as a data scientist at a
cyber security startup, and the company wants to provide the clients with a
tool to automatically detect phishing attempts performed through emails and
warn clients about them. The idea is to scan emails and determine for each
web URL they include whether it points to a phishing-related web page or not.

Since phishing attempts are an always-adapting efforts, static black lists or
white lists composed of good or bad URLs seen in the past are simply not
enough to make a good filtering system for the future. The way the company
chose to deal with this challenge is to have you train a Machine Learning
model to generalize what a phishing URL looks like from historic data!

To enable you to do this the company's security team has collected a set of
benign (meaning OK, or Kosher) URLs and phishing URLs observed during 2019
(not necessarily in clients emails). They have also wrote a script
extracting features they believe should help discern phishing URLs from
benign ones.

These features are divided to three sub-sets:

* String Characteristics - Extracted from the URL string itself.
* Domain Characteristics - Extracted by interacting with the domain provider.
* Web Page Characteristics - Extracted from the content of the web page the
  URL points to.

The string characteristics are based the way URLs are structured, and what
their different parts do. Here is an informative illustration. You can read
more at Mozilla's `What is a URL
<https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL>`__
article. We'll see the specific features soon.
"""

#%%

from IPython.core.display import HTML
from IPython.display import Image

Image(url= "https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL/mdn-url-all.png")

#%%
# (Note: This is a slightly synthetic dataset based on `a great project
# <https://github.com/Rohith-2/url_classification_dl>`__ by `Rohith Ramakrishnan
# <https://www.linkedin.com/in/rohith-ramakrishnan-54094a1a0/>`__ and others,
# accompanied by a `blog post
# <https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d>`__.
# The authors has released it under an open license per our request, and for that
# we are very grateful to them.)

#%%
# **Installing requirements**
#
# .. code:: python
#
#     import sys
#     !{sys.executable} -m pip install deepchecks --quiet

#%%
# Loading the data
# ================
# OK, let's take a look at the data!

import numpy as np
import pandas as pd
import sklearn

import deepchecks

pd.set_option('display.max_columns', 45); SEED=832; np.random.seed(SEED);

#%%

from deepchecks.tabular.datasets.classification.phishing import load_data

#%%

df = load_data(data_format='dataframe', as_train_test=False)

#%%

df.shape

#%%

df.head(5)

#%%
# Here is the actual list of features:

df.columns

#%%
# Feature List
# ------------
# And here is a short explanation of each:
#
# =============  =========================   =======================================================================
# Feature Name   Feature Group               Description 
# =============  =========================   =======================================================================
# target         Meta Features               0 if the URL is benign, 1 if it is related to phishing
# month          Meta Features               The month this URL was first encountered, as an int
# scrape_date    Meta Features               The exact date this URL was first encountered
# ext            String Characteristics      The domain extension
# urlLength      String Characteristics      The number of characters in the URL
# numDigits      String Characteristics      The number of digits in the URL
# numParams      String Characteristics      The number of query parameters in the URL
# num_%20        String Characteristics      The number of '%20' substrings in the URL
# num_@          String Characteristics      The number of @ characters in the URL
# entropy        String Characteristics      The entropy of the URL
# has_ip         String Characteristics      True if the URL string contains an IP addres
# hasHttp        Domain Characteristics      True if the url's domain supports http
# hasHttps       Domain Characteristics      True if the url's domain supports https
# urlIsLive      Domain Characteristics      The URL was live at the time of scraping
# dsr            Domain Characteristics      The number of days since domain registration
# dse            Domain Characteristics      The number of days since domain registration expired
# bodyLength     Web Page Characteristics    The number of charcters in the URL's web page
# numTitles      Web Page Characteristics    The number of HTML titles (H1/H2/...) in the page
# numImages      Web Page Characteristics    The number of images in the page
# numLinks       Web Page Characteristics    The number of links in the page
# specialChars   Web Page Characteristics    The number of special characters in the page
# scriptLength   Web Page Characteristics    The number of charcters in scripts embedded in the page
# sbr            Web Page Characteristics    The ratio of scriptLength to bodyLength (`= scriptLength / bodyLength`)
# bscr           Web Page Characteristics    The ratio of bodyLength to specialChars (`= specialChars / bodyLength`)
# sscr           Web Page Characteristics    The ratio of scriptLength to specialChars (`= scriptLength / specialChars`)
# =============  =========================   =======================================================================

#%%
# Data Integrity with Deepchecks!
# ===============================
# The nice thing about the ``deepchecks`` package is that we can already use it out 
# of the box! Instead of running a single check, we use a pre-defined test suite to
# run a host of data validation checks.
#
# We think it's valuable to start off with these types of suites as there are various
# issues we can identify at the get go just by looking at raw data.
#
# We will first import the appropriate factory function from the ``deepchecks.suites``
# module - in this case, an integrity suite tailored for a single dataset (as
# opposed to a division into a train and test, for example) - and use it to create a
# new suite object:

from deepchecks.tabular.suites import data_integrity

integ_suite = data_integrity()

#%%
# We will now run that suite on our data. While running on the native DataFrame is possible in some cases, it is
# recommended to wrap it with the ``deepchecks.tabular.Dataset`` object instead, to give
# the package a bit more context, namely what is the label column, and whether
# we have a datetime column (we have, as an index, so we'll set
# ``set_datetime_from_dataframe_index=True``), or any categorical features (we have
# none after one-hot encoding them, so we'll set ``cat_features=[]`` explicitly).

dataset = deepchecks.tabular.Dataset(df=df, label='target',
                                     set_datetime_from_dataframe_index=True, cat_features=[])
integ_suite.run(dataset)

#%%
# Understanding the checks' results!
# ==================================
# Ok, so we've got some interesting results! Even though this is quite a tidy dataset
# without even any preprocessing, ``deepchecks`` has found a couple of columns
# (``has_ip`` and ``urlIsLive``) containing only a single value and a couple of
# duplicate values.
#
# We also get a nice list of all checks that turned out ok, and what each check is about.
#
# So nothing dramatic, but we will be sure to drop those useless columns. :)
#
# Preprocessing
# =============
# Let's split the data to train and test first. Since we want to examine how well a model
# can generalize from the past to the future, we'll simply assign the first months of
# the dataset to the training set, and the last few months to the test set.

raw_train_df = df[df.month <= 9]
len(raw_train_df)

#%%

raw_test_df = df[df.month > 9]
len(raw_test_df)

#%%
# Ok! Let's process the data real quick and see how some baseline classifiers perform!
#
# We'll just set the scrape date as our index, drop a few useless columns, one-hot
# encode our categorical ext column and scale all numeric data:

from deepchecks.tabular.datasets.classification.phishing import \
    get_url_preprocessor

pipeline = get_url_preprocessor()

#%%
# Now we'll fit on and transform the raw train dataframe:

train_df = pipeline.fit_transform(raw_train_df)
train_X = train_df.drop('target', axis=1)
train_y = train_df['target']
train_X.head(3)

#%%
# And apply the same fitted preprocessing pipeline (with the fitted scaler, for example)
# to the test dataframe:

test_df = pipeline.transform(raw_test_df)
test_X = test_df.drop('target', axis=1)
test_y = test_df['target']
test_X.head(3)

#%%

from sklearn.linear_model import LogisticRegression; from sklearn.metrics import accuracy_score; hyperparameters = {'penalty': 'l2', 'fit_intercept': True, 'random_state': SEED, 'C': 0.009}

#%%

logreg = LogisticRegression(**hyperparameters)
logreg.fit(train_X, train_y);
pred_y = logreg.predict(test_X)

#%%

accuracy_score(test_y, pred_y)

#%%
# Ok, so we've got a nice accuracy score from the get go! Let's see what deepchecks
# can tell us about our model...

from deepchecks.tabular.suites import train_test_validation

#%%

vsuite = train_test_validation()

#%%
# Now that we have separate train and test DataFrames, we will create two ``deepchecks.tabular.Dataset`` objects to enable
# this suite and the next one to run addressing the train and test dataframes according to their role. Notice that here
# we pass the label as a column instead of a column name, because we've seperated the feature DataFrame from the target.

ds_train = deepchecks.tabular.Dataset(df=train_X, label=train_y, set_datetime_from_dataframe_index=True,
                                      cat_features=[])
ds_test = deepchecks.tabular.Dataset(df=test_X, label=test_y, set_datetime_from_dataframe_index=True, cat_features=[])

#%%
# Now we just have to provide the ``run`` method of the suite object with both the
# model and the ``Dataset`` objects.

vsuite.run(model=logreg, train_dataset=ds_train, test_dataset=ds_test)

#%%
# Understanding the checks' results!
# ==================================
# Whoa! It looks like we have some time leakage!
#
# The ``Conditions`` Summary section showed that the ``Date Train-Test Leakage (overlap)``
# check was the only failed check. The ``Additional Outputs`` section helped us understand
# that the latest date in the train set belongs to January 2020!
#
# It seems some entries from January 2020 made their way into the train set. We assumed
# the ``month`` columns was enough to split the data with (which it would, have all data
# was indeed from 2019), but as in real life, things were a bit messy. We'll adjust our
# preprocessing real quick, and with methodological errors out of the way we'll get to
# checking our model's performance.
#
# it is also worth mentioning that deepchecks found that ``urlLength`` is the only
# feature that alone can predict the target with some measure of success. This is
# worth investigating!
#
# Adjusting our preprocessing and refitting the model
# ---------------------------------------------------
# Let's just drop any row from 2020 from the raw dataframe and take it all from there

df = df[~df['scrape_date'].str.contains('2020')]
df.shape

#%%

pipeline = get_url_preprocessor()

#%%

train_df = pipeline.fit_transform(raw_train_df)
train_X = train_df.drop('target', axis=1)
train_y = train_df['target']
train_X.head(3)

#%%

test_df = pipeline.transform(raw_test_df)
test_X = test_df.drop('target', axis=1)
test_y = test_df['target']
test_X.head(3)

#%%

logreg.fit(train_X, train_y)

#%%

pred_y = logreg.predict(test_X)

#%%

accuracy_score(test_y, pred_y)

#%%
# Deepchecks' Performance Checks
# ==============================
# Ok! Now that we're back on track lets run some performance checks to see how we did.

from deepchecks.tabular.suites import model_evaluation

#%%

msuite = model_evaluation()

#%%

ds_train = deepchecks.tabular.Dataset(df=train_X, label=train_y, set_datetime_from_dataframe_index=True, cat_features=[])
ds_test = deepchecks.tabular.Dataset(df=test_X, label=test_y, set_datetime_from_dataframe_index=True, cat_features=[])

#%%

msuite.run(model=logreg, train_dataset=ds_train, test_dataset=ds_test)

#%%
# Understanding the checks' results!
# ==================================
# Ok! Now that we're back on track lets run some performance checks to see how we did.
#
# * ``Simple Model Comparison`` - This checks make sure our model outperforms a very
#   simple model to some degree. Having it fail means we might have a serious problem.
# * ``Model Error Analysis`` - This check analyses model errors and tries to find a
#   way to segment our data in a way that is informative to error analysis. It seems
#   that it found a valuable way to segment our data, error-wise, using the ``urlLength``
#   feature. We'll look into it soon enough.
#
# Looking at the metric plots for F1 for both our model and a simple one we see their
# performance are almost identical! How can this be? Fortunately the confusion
# matrices automagically generated for both the training and test sets help us
# understand what has happened.
#
# Our evidently over-regularized classifier was over-impressed by the majority class
# (0, or non-malicious URL), and predicted a value of 0 for almost all samples in
# both the train and the test set, which yielded a seemingly-impressive 97% accuracy
# on the test set just due to the imbalanced nature of the problem.
#
# ``deepchecks`` also generated plots for F1, precision and recall on both the train
# and test set, as part of the performance report, and these also help us see
# recall scores are almost zero for both sets and understand what happened.
#
# Trying out a different classifier
# =================================
# So let's throw something a bit more rich in expressive power at the problem - a decision tree!

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=SEED)
model.fit(train_X, train_y)
msuite.run(model=model, train_dataset=ds_train, test_dataset=ds_test)

#%%
# Boosting our model!
# ===================
# To try and solve the overfitting issue let's try and throw at a problem an ensemble
# model that has a bit more resilience to overfitting than a decision tree: a
# gradient-boosted ensemble of them!

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=250, random_state=SEED, max_depth=20, subsample=0.8 , loss='exponential')
model.fit(train_X, train_y)
msuite.run(model=model, train_dataset=ds_train, test_dataset=ds_test)

#%%
# Understanding the checks' results!
# ==================================
# Again, ``deepchecks`` supplied some interesting insights, including a considerable
# performance degradation between the train and test sets. We can see that the
# degradation in performance between the train and test set that we witnessed before
# was mitigated only very little.
#
# However, for a boosted model we get a pretty cool *Boosting Overfit* check that
# plots the accuracy of the model along increasing boosting iterations of the model.
# This can help us see that we might have a minor case of overfitting here, as train
# set accuracy is achieved rather early on, and while test set performance improve
# for a little while longer, they show some degradation starting from iteration 135.
#
# This at least points to possible value in adjusting the ``n_estimators``
# parameter, either reducing it or increasing it to see if degradation continues or
# perhaps the trends shifts.
#
# Wrapping it all up!
# ===================
# We haven't got a decent model yet, but ``deepchecks`` provides us with numerous
# tools to help us navigate our development and make better feature engineering
# and model selection decisions, by easily making critical issues in data drift,
# overfitting, leakage, feature importance and model calibration readily accessible.
#
# And this is just what ``deepchecks`` can do out of the box, with the prebuilt
# checks and suites! There is a lot more potential in the way the package lends
# itself to easy customization and creation of checks and suites tailored to your
# needs. We will touch upon some such advanced uses in future guides.
#
# We, however, hope this example can already provide you with a good starting point
# for getting some immediate benefit out of using deepchecks! Have fun, and reach
# out to us if you need assistance! :)
