# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Runs extensive tests for all checks in suites using the datasets & models in deepchecks-extensive-test-assets"""
import time
import boto3
from io import BytesIO
import sys
import pprint

from deepchecks import Suite
from deepchecks.suites import OverallSuite
from deepchecks.base import Dataset
import pandas as pd
import json
import joblib


def has_errors(d):
    if isinstance(d, dict):
        if len(d) == 0:
            return False
        return any(has_errors(v) for v in d.values())
    else:
        return True


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
    error_log = {}
    for dataset in datasets:
        run_time[dataset] = {}
        displayed_results[dataset] = {}
        error_log[dataset] = {}
        train_df = pd.read_csv(f's3://{bucket_name}/{dataset}/train.csv')
        test_df = pd.read_csv(f's3://{bucket_name}/{dataset}/val.csv')
        metadata = json.loads(s3.Object(bucket_name, f'{dataset}/metadata.json').get()['Body'].read().decode('utf-8'))
        train_ds = Dataset(train_df, label_name=metadata['label_name'], features=metadata['features'],
                           cat_features=metadata['cat_features'])
        test_ds = Dataset(test_df, label_name=metadata['label_name'], features=metadata['features'],
                          cat_features=metadata['cat_features'])
        models = list(filter(lambda x: x.key.endswith('joblib'), list(S3_BUCKET.objects.filter(Prefix=dataset).all())))
        for model_obj in models:
            model_file = model_obj.get()
            model = joblib.load(BytesIO(model_file['Body'].read()))
            model_name = model_obj.key.split('/')[1].split('.')[0]
            displayed_results[dataset][model_name] = {}
            run_time[dataset][model_name] = {}
            error_log[dataset][model_name] = {}
            print(f'Running dataset {dataset} model {model_name}')
            for name, check in OverallCheckSuite.checks.items():
                start_t = time.time()
                # Run check as suite so that MLChecksValueError are not captured as errors
                suite_of_check_to_run = Suite('Test suite', check)
                check_name = check.__class__.__name__
                try:
                    displayed_results[dataset][model_name][check_name] = suite_of_check_to_run.run(model, train_ds,
                                                                                                   test_ds, 'both')
                except Exception as e:
                    error_log[dataset][model_name][check_name] = str(e)
                end_t = time.time()
                run_time[dataset][model_name][check_name] = end_t - start_t

    if has_errors(error_log):
        sys.exit(pprint.pformat(error_log))
