from datetime import datetime

import joblib
import pandas as pd
from airflow.decorators import dag, task, short_circuit_task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


@dag(dag_id='model_training_with_deepchecks_validation',
     schedule_interval=None,
     default_args={
        'owner': 'airflow',
        'start_date': datetime(2023, 1, 1),
     },
     params={
         'bucket': 'deepchecks-public',
         'data_key': 'matan/data.csv',
         'train_path': 'train.csv',
         'test_path': 'test.csv',
         'model_path': 'model.joblib'
     },
     catchup=False)
def model_training_dag():

    @short_circuit_task
    def validate_data(**context):
        from deepchecks.tabular.suites import data_integrity
        from deepchecks.tabular import Dataset

        hook = S3Hook('aws_connection')
        file_name = hook.download_file(key=context['params']['data_key'], bucket_name=context['params']['bucket'],
                                       local_path='.')
        data_df = pd.read_csv(file_name)
        dataset = Dataset(data_df, label='label', cat_features=[])
        suite_result = data_integrity().run(dataset)
        suite_result.save_as_html('data_validation.html')
        hook.load_file(
            filename='data_validation.html',
            key='results/data_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        context['ti'].xcom_push(key='data', value=file_name)
        return suite_result.passed()

    @short_circuit_task
    def validate_train_test_split(**context):
        from deepchecks.tabular.suites import train_test_validation
        from deepchecks.tabular import Dataset

        data = pd.read_csv(context['ti'].xcom_pull(key='data'))
        train_df, test_df = data.iloc[:len(data) // 2], data.iloc[len(data) // 2:]
        train_df.to_csv(context['params']['train_path'])
        test_df.to_csv(context['params']['test_path'])

        train = Dataset(train_df, label='label', cat_features=[])
        test = Dataset(test_df, label='label', cat_features=[])
        suite_result = train_test_validation().run(train_dataset=train, test_dataset=test)
        suite_result.save_as_html('split_validation.html')
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename='split_validation.html',
            key='results/split_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        return suite_result.passed()

    @task
    def train_model(**context):
        train_df = pd.read_csv(context['params']['train_path'])
        # Train model and upload to s3
        model = ...

        joblib.dump(model, context['params']['model_path'])
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename=context['params']['model_path'],
            key='results/model.joblib',
            bucket_name=context['params']['bucket'],
            replace=True
        )

    @task
    def validate_model_performance(**context):
        from deepchecks.tabular.suites import model_evaluation
        from deepchecks.tabular import Dataset

        train_df = pd.read_csv(context['params']['train_path'])
        test_df = pd.read_csv(context['params']['test_path'])
        model = joblib.load(context['params']['model_path'])

        train = Dataset(train_df, label='label', cat_features=[])
        test = Dataset(test_df, label='label', cat_features=[])
        suite_result = model_evaluation().run(train_dataset=train, test_dataset=test, model=model)
        suite_result.save_as_html('model_validation.html')
        hook = S3Hook('aws_connection')
        hook.load_file(
            filename='model_validation.html',
            key='results/model_validation.html',
            bucket_name=context['params']['bucket'],
            replace=True
        )
        return suite_result.passed()

    validate_data() >> validate_train_test_split() >> train_model() >> validate_model_performance()

model_training_dag()
