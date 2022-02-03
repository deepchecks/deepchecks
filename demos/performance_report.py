import pandas as pd
import plotly.express as px

from deepchecks import CheckResult
from deepchecks.vision.base import SingleDatasetCheck


class MeanAveragePrecisionReport(SingleDatasetCheck):

    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        results = pd.DataFrame(columns=['Area size', 'mAP(COCO challenge)', 'AP_50', 'AP_75'])
        results.loc[0] = ['All', 34.9, 55.7, 37.4]
        results.loc[1] = ['Small (area<32^2)', 15.6, 30.4, 17.9]
        results.loc[2] = ['Medium (32^2<area<96^2)', 38.7, 60.1, 42.7]
        results.loc[3] = ['Large (area<96^2)', 50.9, 92.6, 53.4]

        data = {
            'IoU': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'AP': [55.7, 50.4, 46.5, 40.1, 38.5, 37.4, 30.5, 24.2, 19.4, 8.8]
        }
        df = pd.DataFrame.from_dict(data)

        fig = px.line(df, x="IoU", y="AP", title='AP over increasing IoU thresholds')

        return CheckResult(value=results, display=[results, fig])

    def __init__(self, prediction_formatter):
        super().__init__()
        self.prediction_formatter = prediction_formatter

    def run(self, dataset, model=None) -> CheckResult:
        return self.run_logic(None)

