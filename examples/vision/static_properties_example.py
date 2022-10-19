import pandas as pd
import numpy as np
from deepchecks.vision.checks import ImagePropertyOutliers
from deepchecks.vision.datasets.detection.coco import load_dataset
from deepchecks.vision.utils import static_properties_from_df

train_data = load_dataset(train=True, object_type='VisionData')

# Say we have a dataframe with previously calculated properties
df = pd.DataFrame({'property1': np.random.random(train_data.num_samples),
                   'property2': np.random.random(train_data.num_samples)})

static_props = static_properties_from_df(df, image_cols=('property1', 'property2'))


check = ImagePropertyOutliers()
result = check.run(train_data, train_properties=static_props)
