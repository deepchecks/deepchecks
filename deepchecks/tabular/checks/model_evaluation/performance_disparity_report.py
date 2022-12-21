from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Dataset
from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotImplementedError
from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind
from deepchecks.utils.typing import Hashable
from typing import Union, Callable
from deepchecks.utils.performance.partition import partition_column

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import itertools


class PerformanceDisparityReport(SingleDatasetCheck):
    """
    Check for performance disparities across subgroups, while optionally accounting for a control variable.

    The check compares performance scores within subgroups defined by a given feature to average performance scores. Subgroups with large performance disparities are displayed. Conditions can be added to flag performance disparities above a given threshold. If a control feature is provided, then the data is first split across the groups defined by this feature. Then, subgroup performance is compared to average performance at the corresponding control feature level.

    Parameters
    ----------
    feature : Hashable
        Feature evaluated for performance disparities. Numerical features are binned into `max_segments` quantiles. Categorical features are not transformed.
    control_feature : Hashable, default: None
        Feature used to group data prior to evaluating performance disparities (disparities are only assessed within the groups defined by this feature). Numerical features are binned into `max_segments` quantiles. Categorical features are re-grouped into at most `max_segments` groups if necessary.
    scorer : str, default: None
        Name of the performance score function to use.
    max_segments : int, default: 10
        Maximum number of segments into which `control_feature` is binned.
    min_subgroup_size : int, default: 5
        Minimum size of a subgroup for which to compute a performance score.
    max_subgroups_to_display : int, default: 5
        Maximum number of subgroups to display in the "largest performance disparity" plot.
    use_avg_defaults : bool, default: True
        If no scorer was provided, determines whether to return an average score (if True) or a score per class (if False).
    n_samples : int, default: 1_000_000
        Number of samples from the dataset to use.
    random_state : int, default: 42
        Random state to use for probability sampling.
    """

    def __init__(
        self,
        protected_feature: Hashable,
        control_feature: Hashable = None,
        scorer: str = None,  # TODO: Allow for Callable or DeepCheckScorer
        max_segments: int = 10,
        min_subgroup_size: int = 5,
        max_subgroups_to_display: int = 5,
        use_avg_defaults: bool = True,
        n_samples: int = 1_000_000,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.protected_feature = protected_feature
        self.control_feature = control_feature
        self.max_segments = max_segments
        self.min_subgroup_size = min_subgroup_size
        self.scorer = scorer
        self.max_subgroups_to_display = max_subgroups_to_display
        self.use_avg_defaults = use_avg_defaults
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """
        Returns
        -------
        CheckResult
            value is a dataframe with performance scores for within each subgroup defined by `feature` and average scores across these subgroups. If `control_feature` was provided, then performance scores are further disaggregated by the gruops defined by this feature.
            display is a Figure showing the subgroups with the largest performance disparities.
        """
        model = context.model
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        if self.scorer is not None:
            scorer = context.get_single_scorer({self.scorer: self.scorer}, use_avg_defaults=self.use_avg_defaults)
        else:
            scorer = context.get_single_scorer(use_avg_defaults=self.use_avg_defaults)

        partitions = self.make_partitions(dataset)

        scores_df = self.make_scores_df(model, dataset, scorer, partitions)

        if context.with_display:
            fig = self.make_largest_difference_figure2(scores_df)
        else:
            fig = None

        return CheckResult(value=scores_df, display=fig)

    def make_partitions(self, dataset):
        """
        Define partitions of a given dataset based on `feature` and `control_feature`
        """
        partitions = {}

        if dataset.is_categorical(self.protected_feature):
            partitions[self.protected_feature] = partition_column(dataset, self.protected_feature, max_segments=np.Inf)
        else:
            partitions[self.protected_feature] = partition_column(
                dataset, self.protected_feature, max_segments=self.max_segments
            )
        if self.control_feature is not None:
            partitions[self.control_feature] = partition_column(
                dataset, self.control_feature, max_segments=self.max_segments
            )

        return partitions

    def make_scores_df(self, model, dataset, scorer, partitions):
        """
        Computes performance scores disaggregated by `feature` and `control_feature` levels, and averaged over `feature` for each `control_feature` level. Also computes subgroup size.
        """
        classwise = is_classwise(scorer, model, dataset)

        scores_df = expand_grid(**partitions, _scorer=[scorer])

        scores_df["_dataset"] = scores_df.apply(lambda x: combine_filters(x[partitions.keys()], dataset.data), axis=1)
        scores_df["_score"] = scores_df.apply(lambda x: x._scorer(model, dataset.copy(x._dataset)), axis=1)
        if self.control_feature is not None:
            control_scores = {
                x.label: scorer(model, dataset.copy(x.filter(dataset.data)))
                for x in scores_df[self.control_feature].unique()
            }
            scores_df["_baseline"] = scores_df.apply(lambda x: control_scores[x[self.control_feature].label], axis=1)
        else:
            overall_score = scorer(model, dataset)
            scores_df["_baseline"] = scores_df.apply(lambda x: overall_score, axis=1)
        scores_df["_scorer"] = scores_df.apply(lambda x: x._scorer.name, axis=1)
        scores_df["_count"] = scores_df.apply(lambda x: len(x._dataset), axis=1)
        for col_name in partitions.keys():
            scores_df[col_name] = scores_df.apply(lambda x: x[col_name].label, axis=1)

        scores_df.drop(labels=["_dataset"], axis=1, inplace=True)

        if classwise:
            scores_df.insert(len(scores_df.columns) - 3, "_class", scores_df.apply(lambda x: list(x["_score"]), axis=1))
            scores_df["_score"] = scores_df.apply(lambda x: list(x["_score"].values()), axis=1)
            scores_df["_baseline"] = scores_df.apply(lambda x: list(x["_baseline"].values()), axis=1)
            scores_df = scores_df.explode(column=["_score", "_class", "_baseline"])

        scores_df["_score"] = scores_df["_score"].astype(float)
        scores_df["_baseline"] = scores_df["_baseline"].astype(float)

        scores_df = scores_df.loc[scores_df._count >= self.min_subgroup_size]

        return scores_df

    def make_visual_df(self, scores_df: pd.DataFrame):
        """
        Make visualization-ready dataframe.
        """
        visual_df = scores_df.copy()
        visual_df["_diff"] = visual_df["_score"] - visual_df["_baseline"]
        visual_df["_group"] = visual_df[self.protected_feature]
        visual_df["_avg_group"] = "All"
        if self.control_feature is not None:
            visual_df["_group"] += " (" + visual_df[self.control_feature] + ")"
            visual_df["_avg_group"] = visual_df[self.control_feature]
        if "_class" in visual_df.columns:
            visual_df["_group"] += ", " + visual_df["_class"].astype(str)
        visual_df["_count"] = "(" + visual_df._count.astype(str) + ")"
        class_string = ""
        if "_class" in scores_df.columns:
            class_string = ", " + scores_df._class.astype(str)
        control_feature_string = ""
        if self.control_feature is not None:
            control_feature_string = visual_df[self.control_feature]

        visual_df["_details"] = (
            'Score difference between average "'
            + control_feature_string
            + class_string
            + '" ('
            + visual_df["_baseline"].round(3).astype(str)
            + ') and "'
            + visual_df["_group"]
            + '" ('
            + visual_df._score.round(3).astype(str)
            + ")."
        )
        visual_df = visual_df.sort_values(by="_diff", ascending=True)
        visual_df = visual_df.head(self.max_subgroups_to_display)
        visual_df = visual_df.sort_values(by="_diff", ascending=False)

        return visual_df

    def make_largest_difference_figure(self, scores_df: pd.DataFrame):
        """
        Create "largest performance disparity" figure.

        Parameters
        ----------
        scores_df : DataFrame
            Dataframe of performance scores, as returned by `make_scores_df()`, disaggregated by feature and control_feature, and with average scores for each control_feature level. Columns named after `feature` and (optionally) `control_feature` are expected, as well as columns named "_scorer", "_score", "_baseline", and "_count".

        Returns
        -------
        Figure
            Figure showing subgroups with the largest performance disparities.
        """

        title = f"Largest differences between average score and subgroup score"
        if self.control_feature is not None:
            title += f" at a given {self.control_feature} level"

        visual_df = self.make_visual_df(scores_df)

        title += f"<br><sup>(Group size indicated in parenthesis)</sup>"
        fig = px.bar(
            visual_df,
            x="_diff",
            y="_group",
            barmode="group",
            text="_count",
            hover_data=["_details"],
            title=title,
            labels={"_diff": "Score difference", "_group": "Group"},
        )
        return fig

    def make_largest_difference_figure2(self, scores_df: pd.DataFrame):
        """
        Create "largest performance disparity" figure, version 2.

        Parameters
        ----------
        scores_df : DataFrame
            Dataframe of performance scores, as returned by `make_scores_df()`, disaggregated by feature and control_feature, and with average scores for each control_feature level. Columns named after `feature` and (optionally) `control_feature` are expected, as well as columns named "_scorer", "_score", "_baseline", and "_count".

        Returns
        -------
        Figure
            Figure showing subgroups with the largest performance disparities.
        """
        visual_df = self.make_visual_df(scores_df)

        fig = go.Figure()

        for _, row in visual_df.iterrows():
            x = row["_group"]
            y0 = row["_baseline"]
            y1 = row["_score"]
            color = "orangered" if row["_diff"] < 0 else "limegreen"
            textposition = ["top left", "top right"] if row["_diff"] < 0 else ["top right", "top left"]

            baseline_pt_label = "baseline"
            if self.control_feature is not None:
                baseline_pt_label = "(" + row[self.control_feature] + ")"
            if "_class" in visual_df.columns:
                baseline_pt_label += ", " + row["_class"]
            fig.add_traces(
                go.Scatter(
                    x=[y1, y0],
                    y=[x, x],
                    text=[row._group, baseline_pt_label],
                    textposition=textposition,
                    textfont=dict(size=11),
                    marker=dict(
                        color=["white", "#222222"], 
                        symbol=0, 
                        size=7, 
                        line=dict(width=[3, 3], 
                        color=[color, color])
                    ),
                    line=dict(color=color, width=10),
                    opacity=0.9,
                    showlegend=False,
                    hoverinfo=["x+y", "x+text"],
                    customdata=visual_df._group,
                    mode="lines+text+markers",
                    cliponaxis=False,
                ),
            )

        title = f"Top {len(visual_df)} largest differences between baseline score and subgroup score"
        xaxis_title = f"{self.protected_feature}"
        if self.control_feature is not None:
            title += f" (controlling for {self.control_feature})"
            title += (
                f"<br><sup>(Black point is the baseline score for the corresponding {self.control_feature} group)</sup>"
            )
            xaxis_title += f" ({self.control_feature})"
        else:
            title += f"<br><sup>(Black point is the baseline score)</sup>"
        if "_class" in visual_df.columns:
            xaxis_title += ", model_class"

        fig.update_layout(
            title=title,
            yaxis_title=xaxis_title,
            xaxis_title=f"Score",
            yaxis_showticklabels=False,
            yaxis_showgrid=False,
        )

        return fig

    def add_condition_bounded_performance_difference(self, lower_bound, upper_bound=np.Inf):
        """Add condition - require performance difference between baselines and subgroups to be within the given bounds.

        Parameters
        ----------
        lower_bound : float
            Lower bound on (score - baseline). 
        upper_bound : float, default: Infinity
            Upper bound on (score - baseline). Infinite by default (large scores do no trigger the condition).
        """
        def bounded_performance_difference_condition(scores_df: pd.DataFrame) -> ConditionResult:
            differences = scores_df["_score"] - scores_df["_baseline"]
            I = (differences < lower_bound) | (differences > upper_bound)

            details = f'Found {sum(I)} subgroups with performance differences outside of the given bounds.'
            category = ConditionCategory.PASS if sum(I) == 0 else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Performance differences are bounded between {lower_bound} and {upper_bound}.',
                                  bounded_performance_difference_condition)
    
    def add_condition_bounded_relative_performance_difference(self, lower_bound, upper_bound=np.Inf):
        """Add condition - require relative performance difference between baselines and subgroups to be within the given bounds.

        Parameters
        ----------
        lower_bound : float
            Lower bound on (score - baseline)/baseline. 
        upper_bound : float, default: Infinity
            Upper bound on (score - baseline)/baseline. Infinite by default (large scores do no trigger the condition).
        """
        def bounded_performance_difference_condition(scores_df: pd.DataFrame) -> ConditionResult:
            differences = scores_df["_score"] - scores_df["_baseline"]
            I_zero = scores_df["_baseline"] == 0
            differences[I_zero] = differences * np.Inf
            differences[~I_zero] = differences[~I_zero]/scores_df["_baseline"][~I_zero]
            I = (differences < lower_bound) | (differences > upper_bound)

            details = f'Found {sum(I)} subgroups with relative performance differences outside of the given bounds.'
            category = ConditionCategory.PASS if sum(I) == 0 else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Relative performance differences are bounded between {lower_bound} and {upper_bound}.',
                                  bounded_performance_difference_condition)


def expand_grid(**kwargs):
    """
    Create a dataframe with one column for each named argument and rows corresponding to all possible combinations of the given arguments.
    """
    return pd.DataFrame.from_records(itertools.product(*kwargs.values()), columns=kwargs.keys())


def combine_filters(filters, dataframe):
    """
    Combine segment filters

    Parameters
    ----------
    filters: Series
        Series indexed by segment names and with values corresponding to segment filters to be applied to the data.
    dataframe: DataFrame
        DataFrame to which filters are applied.

    Returns
    -------
    DataFrame
        Data filtered to the given combination of segments.
    """
    segments = filters.index.values
    filtered_data = filters[segments[0]].filter(dataframe)
    if len(segments) > 1:
        for i in range(1, len(segments)):
            filtered_data = filters[segments[i]].filter(filtered_data)

    return filtered_data


def is_classwise(scorer, model, dataset):
    """
    Check whether a given scorer provides an average score or a score for each class.
    """
    test_result = scorer(model, dataset.copy(dataset.data.head(5)))
    return isinstance(test_result, dict)
