"""Runs extensive tests for all checks in suites using the datasets & models in deepchecks-extensive-test-assets"""
import time
import boto3
from io import BytesIO
from mlchecks.suites import OverallGenericCheckSuite
from mlchecks.base import Dataset
import pandas as pd
import json
import joblib


if __name__ == "__main__":

    bucket_name = "deepchecks-extensive-test-assets"
    s3 = boto3.resource("s3")
    s3_client = boto3.client('s3')
    S3_BUCKET = s3.Bucket(bucket_name)
    session = boto3.Session()

    datasets = list(map(lambda x: x['Prefix'][:-1],
                        s3_client.list_objects(Bucket=bucket_name, Prefix='', Delimiter='/')['CommonPrefixes']))

    displayed_results = {}
    run_time = {}
    for dataset in datasets:
        run_time[dataset] = {}
        displayed_results[dataset] = {}
        train_df = pd.read_csv(f's3://{bucket_name}/{dataset}/train.csv')
        val_df = pd.read_csv(f's3://{bucket_name}/{dataset}/val.csv')
        metadata = json.loads(s3.Object(bucket_name, f'{dataset}/metadata.json').get()['Body'].read().decode('utf-8'))
        train_ds = Dataset(train_df, label=metadata['label_name'], cat_features=metadata['cat_features'])
        val_ds = Dataset(val_df, label=metadata['label_name'], cat_features=metadata['cat_features'])
        models = list(filter(lambda x: x.key.endswith('joblib'),  list(S3_BUCKET.objects.filter(Prefix=dataset).all())))
        for model_obj in models:
            model_file = model_obj.get()
            model = joblib.load(BytesIO(model_file['Body'].read()))
            model_name = model_obj.key.split('/')[1].split('.')[0]
            print(f'Running dataset {dataset} model {model_name}')
            start_t = time.time()
            displayed_results[dataset][model_name] = OverallGenericCheckSuite.run(model, train_ds, val_ds)
            end_t = time.time()
            run_time[dataset][model_name] = end_t - start_t
