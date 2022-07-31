from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
from deepchecks.tabular.suites import model_evaluation

ds_train, ds_test = load_data(data_format='Dataset', as_train_test=True)
rf_clf = load_fitted_model()  # trained sklearn RandomForestClassifier
result = model_evaluation().run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
from deepchecks.tabular.suites import model_evaluation
from deepchecks.tabular.feature_importance import calculate_feature_importance

ds_train, ds_test = load_data(data_format='Dataset', as_train_test=True)
rf_clf = load_fitted_model()  # trained sklearn RandomForestClassifier

fi = calculate_feature_importance(rf_clf, ds_train)
train_proba = rf_clf.predict_proba(ds_train.features_columns)
test_proba = rf_clf.predict_proba(ds_test.features_columns)

# In classification, predicted values can be supplied via the y_pred_train, y_pred_test
# arguments or inferred from the probabilities per class.
result = model_evaluation().run(train_dataset=ds_train, test_dataset=ds_test,
            features_importance=fi, y_proba_train=train_proba, y_proba_test=test_proba)
