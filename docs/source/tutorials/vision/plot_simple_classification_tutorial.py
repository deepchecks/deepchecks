"""
================================================
Using Deepchecks Vision With a Few Lines of Code
================================================

Deepchecks Vision is built to validate your data and model, however complex your model and data may be. That
being said, sometime there is no need to write a full-blown
:doc:`ClassificationData </auto_tutorials/vision/plot_classification_tutorial>` or
:doc:`DetectionData </auto_tutorials/vision/plot_detection_tutorial>`. In the case of a simple classification task,
there is quite a few checks that can be run writing only a few lines of code. In this tutorial, we will show you how
to run all checks that do not require a model on a simple classification task.

This is ideal, for example, when receiving a new dataset for a classification task. Running these checks on the dataset
before even starting with training will give you a quick idea of how the dataset looks like and what potential issues
it contains.
"""

#%%
# Defining the data and model
# ===========================
#
# The data is available from the torch library. We will download and extract it to the current directory.

import urllib.request
import zipfile
import os

url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
urllib.request.urlretrieve(url, 'hymenoptera_data.zip')

with zipfile.ZipFile('hymenoptera_data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Rename val folder to test, because the simple classification task expects a test folder.
if not os.path.exists('hymenoptera_data/test'):
    os.rename('hymenoptera_data/val', 'hymenoptera_data/test')

#%%
# Loading a Simple Classification Dataset
# ========================================
#
# A simple classification dataset is an image dataset structured in the following way:
#
#         - root/
#             - train/
#                 - class1/
#                     image1.jpeg
#             - test/
#                 - class1/
#                     image1.jpeg

from deepchecks.vision.simple_classification_data import load_dataset

train_ds = load_dataset('hymenoptera_data', train=True, object_type='VisionData', image_extension='jpg')
test_ds = load_dataset('hymenoptera_data', train=False, object_type='VisionData', image_extension='jpg')

# Running Deepchecks' full suite
# ==============================
# That's it, we have just defined the classification data object and are ready to run the train_test_validation suite:

from deepchecks.vision.suites import train_test_validation

suite = train_test_validation()
result = suite.run(train_ds, test_ds)

#%%
# Observing the results:
# ===========================
# The results can be saved as a html file with the following code:

result.save_as_html('output.html')

#%%
# Or, if working inside a notebook, the output can be displayed directly by simply printing the result object:

result
