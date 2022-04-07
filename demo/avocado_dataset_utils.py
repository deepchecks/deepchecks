import pandas as pd
import numpy as np


#####################################################
#   Simple Utils for Preprocessing and Splitting    #
#####################################################

def get_classification_label_from_avg_price(df):
    EXPENSIVE_AVOCADO_AVG_PRICE = 1.6
    df['IsExpensive'] = df['AveragePrice'] > EXPENSIVE_AVOCADO_AVG_PRICE
    df = df.drop(columns='AveragePrice')
    return df

def drop_unused_columns(df):
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop(columns=['region', 'Date'])
    return df

def organize_df(df):
    df = df.sort_values(by='Date').reset_index(drop=True)
    df = get_classification_label_from_avg_price(df)
    df = drop_unused_columns(df)
    return df

def ohe_for_type_column(df, org_dummies_columns=None):
    # there are only two values for type, getting dummies for all dataframe together
    generated_dummy_columns = None
    if 'type' in df.columns:
        dummies = pd.get_dummies(df['type'])
        generated_dummy_columns = dummies.columns
        if org_dummies_columns is not None:
            columns_only_in_org = set(org_dummies_columns) - set(generated_dummy_columns)
            if columns_only_in_org:
                # append also columns existant in previous
                dummies[list(columns_only_in_org)] = 0
            columns_only_in_result = set(generated_dummy_columns) - set(org_dummies_columns)
            if columns_only_in_result:
                dummies = dummies.drop(columns=list(columns_only_in_result))
            # set order to be similar
            dummies = dummies[org_dummies_columns]
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns='type')
    return df, generated_dummy_columns

def get_train_test_df_from_raw(df, train_frac=0.7):
    # nothing here special to train/test so doing it together
    df = organize_df(df)
    
    # data is chronological, so split should be accordingly
    train_samples = int(len(df)*train_frac)
    train_df = df[:train_samples]
    test_df = df[train_samples:]
    return train_df, test_df


#####################################################
#        Simple Utils for Adding Dirty Data         #
#####################################################

def add_organic_string_mismatch(df):
    # copying since changes are inplace. returning df for consistency
    df_dirty = df.copy(deep=True)
    df_dirty.loc[df[df['type'] == 'organic'].sample(frac=0.18).index,'type'] = 'Organic'
    df_dirty.loc[df[df['type'] == 'organic'].sample(frac=0.01).index,'type'] = 'ORGANIC'
    return df_dirty


def add_duplicates_with_shuffled_labels(df, label_col_name='IsExpensive'):
    frac_to_duplicate = 0.36
    labels = df[label_col_name]
    duplicated_samples = df.sample(frac=frac_to_duplicate)
    # shuffle labels, will likely add ambiguity
    duplicated_samples[label_col_name] = duplicated_samples[label_col_name].sample(frac=1, ignore_index=True)
    df_dup_labels = pd.concat([df, duplicated_samples], ignore_index=True)
    return df_dup_labels
    

def add_data_duplicates(df, frac_to_duplicate=0.156):
    duplicated_samples = df.sample(frac=frac_to_duplicate)
    df_dup = pd.concat([df, duplicated_samples], axis=0, ignore_index=True)
    return df_dup

    
def add_single_value(df):
    new_column_df = df.copy(deep=True)
    new_column_df['Is Ripe'] = True
    return new_column_df


def add_test_sample_leakage_to_train(train_df, test_df):
    train_df = pd.concat([test_df.sample(frac=0.03), train_df], ignore_index=True)
    return train_df


def add_dirty_data_to_single_df(df):
    df = add_organic_string_mismatch(df)
    df = add_duplicates_with_shuffled_labels(df)
    df = add_data_duplicates(df)
    df = add_single_value(df)
    return df


def add_regression_drift_to_test_labels(test_df, label_col_name = 'AveragePrice'):
    test_df[label_col_name] = test_df[label_col_name] + np.random.rand(test_df.shape[0])*test_df[label_col_name].std()*4
    return test_df