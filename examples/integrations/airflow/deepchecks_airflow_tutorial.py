from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
import joblib
import pandas as pd

from deepchecks.tabular.datasets.classification import adult

dir_path = "suite_results"
# For demo only. Replace that with a S3/GCS other than local filesystem
data_path = os.path.join(os.getcwd(), "data")


def load_adult_dataset(**context):
    df_train, df_test = adult.load_data(data_format='Dataframe')

    try:
        os.mkdir(data_path)
    except OSError:
        print("Creation of the directory {} failed".format(dir_path))

    with open(os.path.join(data_path, "adult_train.csv"), "w") as f:
        df_train.to_csv(f, index=False)
        context["ti"].xcom_push(key="train_path", value=os.path.join(data_path, "adult_train.csv"))
    with open(os.path.join(data_path, "adult_test.csv"), "w") as f:
        df_test.to_csv(f, index=False)
        context["ti"].xcom_push(key="test_path", value=os.path.join(data_path, "adult_test.csv"))


def load_adult_model(**context):
    from deepchecks.tabular.datasets.classification.adult import load_fitted_model

    model = load_fitted_model()
    with open(os.path.join(data_path, "adult_model.joblib"), "wb") as f:
        joblib.dump(model, f)

    context["ti"].xcom_push(key="adult_model", value=os.path.join(data_path, "adult_model.joblib"))


def dataset_integrity_step(**context):
    from deepchecks.tabular.suites import data_integrity
    from deepchecks.tabular.datasets.classification.adult import _CAT_FEATURES, _target
    from deepchecks.tabular import Dataset

    adult_train = pd.read_csv(context.get("ti").xcom_pull(key="train_path"))
    adult_test = pd.read_csv(context.get("ti").xcom_pull(key="test_path"))

    ds_train = Dataset(adult_train, label=_target, cat_features=_CAT_FEATURES)
    ds_test = Dataset(adult_test, label=_target, cat_features=_CAT_FEATURES)

    train_results = data_integrity().run(ds_train)
    test_results = data_integrity().run(ds_test)

    try:
        os.mkdir('suite_results')
    except OSError:
        print("Creation of the directory {} failed".format(dir_path))

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_results.save_as_html(os.path.join(dir_path, f'train_integrity_{run_time}.html'))
    test_results.save_as_html(os.path.join(dir_path, f'test_integrity_{run_time}.html'))


def model_evaluation_step(**context):
    from deepchecks.tabular.suites import model_evaluation
    from deepchecks.tabular.datasets.classification.adult import _CAT_FEATURES, _target
    from deepchecks.tabular import Dataset

    adult_model = joblib.load(context.get("ti").xcom_pull(key="adult_model"))
    adult_train = pd.read_csv(context.get("ti").xcom_pull(key="train_path"))
    adult_test = pd.read_csv(context.get("ti").xcom_pull(key="test_path"))
    ds_train = Dataset(adult_train, label=_target, cat_features=_CAT_FEATURES)
    ds_test = Dataset(adult_test, label=_target, cat_features=_CAT_FEATURES)

    evaluation_results = model_evaluation().run(ds_train, ds_test, adult_model)

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_results.save_as_html(os.path.join(dir_path, f'model_evaluation_{run_time}.html'))


with DAG(
        dag_id="deepchecks_airflow_integration",
        schedule_interval="@daily",
        default_args={
            "owner": "airflow",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
            "start_date": datetime(2021, 1, 1),
        },
        catchup=False,
) as dag:
    load_adult_dataset = PythonOperator(
        task_id="load_adult_dataset",
        python_callable=load_adult_dataset
    )

    integrity_report = PythonOperator(
        task_id="integrity_report",
        python_callable=dataset_integrity_step
    )

    load_adult_model = PythonOperator(
        task_id="load_adult_model",
        python_callable=load_adult_model
    )

    evaluation_report = PythonOperator(
        task_id="evaluation_report",
        python_callable=model_evaluation_step
    )

load_adult_dataset >> integrity_report
load_adult_dataset >> load_adult_model >> evaluation_report

