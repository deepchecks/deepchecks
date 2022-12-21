from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksProcessError, DeepchecksValueError
from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind
from deepchecks.utils.typing import Hashable
from deepchecks.utils.performance.partition import partition_column

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import itertools


class PerformanceDisparityReport(SingleDatasetCheck):
    """
    Check for performance disparities across subgroups, while optionally accounting for a control variable.

    The check identifies "performance disparities": large performance difference for a subgroup compared to the baseline performance for the full population.

    Subgroups are defined by the levels of a "protected" feature. Numerical features are first binned into quantiles, while categorical features are preserved as-is. The baseline score is the overall score when all subgroups are combined. You can add conditions to flag performance differences outside of given bounds. A visual display is also provided to identify subgroups with the largest performance disparities.

    Additionally, the analysis may be separated across the levels of a "control" feature. Numerical features are binned and categorical features are re-binned into `max_number` categories. To account for the control feature, baseline scores and subgroup scores are be computed within each of its levels.

    Parameters
    ----------
    protected_feature : Hashable
        Feature evaluated for performance disparities. Numerical features are binned into `max_segments` quantiles. Categorical features are not transformed.
    control_feature : Hashable, default: None
        Feature used to group data prior to evaluating performance disparities (disparities are only assessed within the groups defined by this feature). Numerical features are binned into `max_segments` quantiles. Categorical features are re-grouped into at most `max_segments` groups if necessary.
    scorer : str, default: None
        Name of the performance score function to use.
    max_segments : int, default: 10
        Maximum number of segments into which `control_feature` is binned.
    min_subgroup_size : int, default: 5
        Minimum size of a subgroup for which to compute a performance score.
    max_subgroups_per_category_to_display : int, default: 3
        Maximum number of subgroups to display.
    max_categories_to_display: int, default: 3
        Maximum number of `control_feature` categories to display.
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
        scorer: str = None,  # TODO: Question: should we allow for Callable or DeepCheckScorer?
        max_segments: int = 10,
        min_subgroup_size: int = 5,
        max_subgroups_per_category_to_display: int = 3,
        max_categories_to_display: int = 3,
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
        self.max_subgroups_per_category_to_display = max_subgroups_per_category_to_display
        self.max_categories_to_display = max_categories_to_display
        self.use_avg_defaults = use_avg_defaults
        self.n_samples = n_samples
        self.random_state = random_state

        self.validate_attributes()

    def validate_attributes(self):
        """
        Validate attributes passed to the check.
        """
        if self.max_segments < 2:
            raise DeepchecksValueError("Maximum number of segments must be at least 2.")

        if self.min_subgroup_size < 1:
            raise DeepchecksValueError("Minimum subgroup size must be at least 1.")

        if self.max_subgroups_per_category_to_display < 1:
            raise DeepchecksValueError("Maximum number of subgroups to display must be at least 1.")

        if self.max_categories_to_display < 1:
            raise DeepchecksValueError("Maximum number of categories to display must be at least 1.")

        if self.n_samples < 1:
            raise DeepchecksValueError("Number of samples must be at least 1.")

        if not isinstance(self.random_state, int):
            raise DeepchecksValueError(f"Random state must be an integer, got {self.random_state}.")

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

        self.validate_run_arguments(dataset.data, model, scorer)

        partitions = self.make_partitions(dataset)

        scores_df = self.make_scores_df(model, dataset, scorer, partitions, context.model_classes)

        if context.with_display:
            fig = self.make_largest_difference_figure(scores_df)
        else:
            fig = None

        return CheckResult(value=scores_df, display=fig)

    def validate_run_arguments(self, data, model, scorer):
        """
        Validate arguments passed to `run_logic` method.
        """
        if self.protected_feature not in data.columns:
            raise DeepchecksValueError(f"Feature {self.protected_feature} not found in dataset.")

        if self.control_feature is not None and self.control_feature not in data.columns:
            raise DeepchecksValueError(f"Feature {self.control_feature} not found in dataset.")

        if self.control_feature is not None and self.control_feature == self.protected_feature:
            raise DeepchecksValueError(f"protected_feature {self.control_feature} and control_feature {self.protected_feature} are the same.")

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

    def make_scores_df(self, model, dataset, scorer, partitions, model_classes):
        """
        Computes performance scores disaggregated by `feature` and `control_feature` levels, and averaged over `feature` for each `control_feature` level. Also computes subgroup size.
        """
        classwise = is_classwise(scorer, model, dataset)

        scores_df = expand_grid(**partitions, _scorer=[scorer])

        scores_df["_dataset"] = scores_df.apply(lambda x: combine_filters(x[partitions.keys()], dataset.data), axis=1)

        def score(data, model, scorer):
            if len(data) == 0:
                if classwise:
                    return {cls: np.nan for cls in model_classes}
                else:
                    return np.nan
            return scorer(model, dataset.copy(data))
        def apply_scorer(x):
            return score(x["_dataset"], model, x["_scorer"])

        scores_df["_score"] = scores_df.apply(apply_scorer, axis=1)
        if self.control_feature is not None:
            control_scores = {
                x.label: score(x.filter(dataset.data), model, scorer) 
                for x in scores_df[self.control_feature].unique()
            }
            control_count = {x.label: len(x.filter(dataset.data)) for x in scores_df[self.control_feature].unique()}
            scores_df["_baseline"] = scores_df.apply(lambda x: control_scores[x[self.control_feature].label], axis=1)
            scores_df["_baseline_count"] = scores_df.apply(
                lambda x: control_count[x[self.control_feature].label], axis=1
            )
        else:
            overall_score = scorer(model, dataset)
            overall_len = len(dataset.data)
            scores_df["_baseline"] = scores_df.apply(lambda x: overall_score, axis=1)
            scores_df["_baseline_count"] = scores_df.apply(lambda x: overall_len, axis=1)
        scores_df["_scorer"] = scores_df.apply(lambda x: x["_scorer"].name, axis=1)
        scores_df["_count"] = scores_df.apply(lambda x: len(x["_dataset"]), axis=1)
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
        
        scores_df["_diff"] = scores_df["_score"] - scores_df["_baseline"]
        scores_df.sort_values("_diff", inplace=True)

        return scores_df

    def _add_differences_traces(self, sub_visual_df, fig, row=1, col=1):
        sub_visual_df = sub_visual_df.sort_values("_diff").head(self.max_subgroups_per_category_to_display)
        sub_visual_df = sub_visual_df.sort_values("_diff", ascending=False)
        for _, df_row in sub_visual_df.iterrows():
            subgroup = df_row[self.protected_feature]
            baseline = df_row["_baseline"]
            score = df_row["_score"]
            color = "orangered" if df_row["_diff"] < 0 else "limegreen"
            extra_label = "<extra></extra>"

            fig.add_trace(
                go.Scatter(
                    x=[score, baseline],
                    y=[subgroup, subgroup],
                    hovertemplate=[
                        "%{y}: %{x} (group size: " + str(df_row["_count"]) + ")" + extra_label,
                        "baseline: %{x} (group size: " + str(df_row["_baseline_count"]) + ")" + extra_label,
                    ],
                    marker=dict(
                        color=["white", "#222222"], symbol=0, size=6, line=dict(width=[2, 2], color=[color, color])
                    ),
                    line=dict(color=color, width=8),
                    opacity=1,
                    showlegend=False,
                    mode="lines+text+markers",
                    cliponaxis=False,
                ),
                row=row,
                col=1,
            )

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

        visual_df = scores_df.copy().dropna()

        has_control = self.control_feature is not None
        has_model_classes = "_class" in visual_df.columns.values

        subplot_grouping = []
        if has_control:
            subplot_grouping += [self.control_feature]
        if has_model_classes:
            subplot_grouping += ["_class"]
        # Get distinct subplot categories with the largest observed differences
        if len(subplot_grouping) > 0:
            subplots_categories = (
                visual_df.sort_values("_diff", ascending=True)[subplot_grouping]
                .drop_duplicates()
                .head(self.max_categories_to_display)
            )
            rows = len(subplots_categories)
            if rows == 0:
                raise DeepchecksProcessError("Nothing to display.")
        else:
            subplots_categories = None
            rows = 1

        subplot_titles = ""
        if has_control:
            subplot_titles += f"{self.control_feature}=" + subplots_categories[self.control_feature]
        if has_control and has_model_classes:
            subplot_titles += f", model_class=" + subplots_categories["_class"]
        if has_model_classes and not has_control:
            subplot_titles = f"model_class=" + subplots_categories["_class"]

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles.values if isinstance(subplot_titles, pd.Series) else None,
            vertical_spacing=0.7 / rows**1.5,
        )

        if subplots_categories is not None:
            i = 0
            for _, cat in subplots_categories.iterrows():
                i += 1
                if has_control and not has_model_classes:
                    I = visual_df[self.control_feature] == cat[self.control_feature]
                elif has_model_classes and not has_control:
                    I = visual_df["_class"] == cat["_class"]
                elif has_control and has_model_classes:
                    I = (visual_df[self.control_feature] == cat[self.control_feature]) & (
                        visual_df["_class"] == cat["_class"]
                    )
                else:
                    raise DeepchecksProcessError(
                        "Cannot use subplot categories without control_feature or model classes."
                    )

                sub_visual_df = visual_df[I]
                self._add_differences_traces(sub_visual_df, fig, row=i, col=1)
        else:
            self._add_differences_traces(visual_df, fig, row=1, col=1)

        title = f"Largest performance differences"
        if has_control and not has_model_classes:
            title += f" within {self.control_feature} categories"
        elif has_model_classes and not has_control:
            title += " model_class categories"
        if has_control and has_model_classes:
            title += f" within {self.control_feature} and model_class categories"

        n_subgroups = len(visual_df[self.protected_feature].unique())
        n_subgroups_shown = min(n_subgroups, self.max_subgroups_per_category_to_display)
        title += f"<br><sup>(Showing {n_subgroups_shown}/{n_subgroups} {self.protected_feature} levels"
        n_cat = 1
        if has_control or has_model_classes:
            n_cat = len(visual_df[subplot_grouping].drop_duplicates())
            title += f" per plot and {rows}/{n_cat} "
            if has_control and not has_model_classes:
                title += f"{self.control_feature}"
            elif has_model_classes and not has_control:
                title += "model_classes"
            else:
                title += f"({self.control_feature}, model_classes)"
            title += " categories"
        title += ")</sup>"

        fig.update_layout(title_text=title)
        fig.update_annotations(x=0, xanchor="left", font_size=12)
        fig.update_layout({f"xaxis{rows}_title": f"{self.scorer} score"})
        fig.update_layout({f"yaxis{i}_title": self.protected_feature for i in range(1, rows + 1)})
        fig.update_layout({f"yaxis{i}_tickmode": "linear" for i in range(1, rows + 1)})

        fig.update_layout(height=150 + 50 * rows + 20 * rows * n_subgroups_shown)

        return fig

    def add_condition_bounded_performance_difference(self, lower_bound, upper_bound=np.Inf):
        """Add condition - require performance difference between baseline and subgroups to be between the given bounds.

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

            details = f"Found {sum(I)} subgroups with performance differences outside of the given bounds."
            category = ConditionCategory.PASS if sum(I) == 0 else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(
            f"Performance differences are bounded between {lower_bound} and {upper_bound}.",
            bounded_performance_difference_condition,
        )


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
    test_result = scorer(model, dataset.copy(dataset.data.head()))
    return isinstance(test_result, dict)
