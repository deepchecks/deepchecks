# load MNIST
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset, load_model
train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')
model = load_model()

# iterate over the dataset
import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
static_preds = []
for vision_data in [train_ds, test_ds]:
    if vision_data is not None:
        static_pred = {}
        for i, batch in enumerate(vision_data):
            predictions = vision_data.infer_on_batch(batch, model, device)
            indexes = list(vision_data.data_loader.batch_sampler)[i]
            static_pred.update(dict(zip(indexes, predictions)))
    else:
        static_pred = None
    static_preds.append(static_pred)
train_preds, tests_preds = static_preds

# pass to the check and view the result
from deepchecks.vision.checks import ClassPerformance
result = ClassPerformance().run(train_ds, test_ds, train_predictions=train_preds, test_predictions=tests_preds)
result.show()
