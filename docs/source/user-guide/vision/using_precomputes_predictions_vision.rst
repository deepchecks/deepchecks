.. _using_precomputed_predictions:

=============================
Using Precomputed Predictions
=============================
Some checks, mainly the ones related to model evaluation, require model predictions in order to run.
In deepchecks, predictions are passed to the suite / check run method in one of the following ways:

* Implementing an infer_on_batch methods in the VisionData object, that allows the checks to compute the predictions.
* Passing the pre-computed predictions as a parameter to the check's run

Passing pre-computed predictions is a simple alternative to using a model in infer_on_batch.
It is specifically recommended to use this option if your model object is unavailable locally (for example if placed on
a separate prediction server) or if the predicting process is computationally expensive or time consuming.

Running Deepchecks with Pre-computed Predictions
------------------------------------------------
