import pandas as pd
from sklearn.datasets import load_iris
from mlchecks.base import Dataset

df = load_iris(return_X_y=False, as_frame=True)
df = pd.concat([df.data, df.target], axis=1)

ds = Dataset(df, index='ssss')
ds = ds.filter_columns_with_validation(columns=['target'])

print(ds.__dict__)