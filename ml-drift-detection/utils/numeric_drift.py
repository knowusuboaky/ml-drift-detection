# ml-drift-detection/numeric_drift.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List


########################## NUMERIC #############################
# ---------------------------------------------------------------------------
# DataFrame subclass for numeric drift summary with interpret() method
# ---------------------------------------------------------------------------
class NumStatsSummary(pd.DataFrame):
    """
    DataFrame for numeric-drift results, with a natural-language summary.

    Extra attributes:
        _p_threshold      – original Levene α
        _corrected_alpha  – Bonferroni-corrected KS α
        _ks_alternative   – 'two-sided' | 'less' | 'greater'
    """
    _metadata = ["_p_threshold", "_corrected_alpha", "_ks_alternative"]

    @property
    def _constructor(self):
        return NumStatsSummary

    def interpret(self) -> str:
        """
        Return a detailed one-sentence summary of all features showing drift,
        including α thresholds and recommendations, with no imperative language.
        """
        αL = getattr(self, "_p_threshold", 0.05)
        αK = getattr(self, "_corrected_alpha", 0.05)
        alt = getattr(self, "_ks_alternative", "two-sided")
        αK_rounded = f"{αK:.4f}"

        drifted = [
            r["Feature"] for _, r in self.iterrows()
            if (float(r["Levene p"]) <= αL) or (float(r["KS p"]) <= αK)
        ]

        if not drifted:
            return (
                "All numeric features examined exhibit distributions that are "
                "statistically indistinguishable from the production benchmark "
                f"at the Levene α-level of {αL} and the Bonferroni-corrected "
                f"KS α-level of {αK_rounded} (alternative = {alt})."
            )

        plural = "feature" if len(drifted) == 1 else "feature(s)"
        feature_list = ", ".join(drifted)

        return (
            f"The {plural} {feature_list} display distributional profiles that "
            "diverge significantly from those observed in the production data, "
            "with at least one of two complementary statistical safeguards "
            f"registering significance—namely, the Levene test for homogeneity "
            f"of variance evaluated at α = {αL}, and the Bonferroni-adjusted "
            f"Kolmogorov–Smirnov test assessed at α = {αK_rounded} under a {alt} "
            "alternative hypothesis—thereby signaling substantive shifts in "
            "spread, central tendency, or overall shape. Prior to promoting this "
            "data to production, it is recommended that the model be retrained on "
            "the updated distribution and that the underlying process be "
            "investigated for potential root causes of the observed drift."
        )

def get_plotly_dist(
    *,
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_cols: List[str],
    mode: str = "dark",
    height: int = 300,
):
    """
    Interactive KDE curves comparing production vs. incoming distributions
    for every numeric feature.

    Parameters
    ----------
    prod_df, new_df : pd.DataFrame
        DataFrames containing the columns in `numeric_cols`.
    numeric_cols : list[str]
        Columns to visualise.
    mode : {"dark", "light"}
        Colour template toggle.
    height : int
        Base height (px) of the figure.

    Returns
    -------
    plotly.graph_objs.Figure
    """

    if mode == "dark":
        bgcolor = "#000"
        textcolor = "#fff"
        new_color = "#ffa726"
        new_fill_color = "rgba(255, 167, 38, 0.5)"
        prod_color = "rgba(255, 165, 0, 0.3)"
        prod_line_color = "#ffa726"
        gridcolor = "#444"
        axis_label_color = "#fff"
        axis_tickfont = dict(color="#fff", size=13, family="Arial")
        axis_titlefont = dict(color="#fff", size=16, family="Arial")
    else:
        bgcolor = "#fff"
        textcolor = "#000"
        new_color = "#d32f2f"                 # strong red
        new_fill_color = "rgba(211,47,47,0.7)"# semi-transparent red
        prod_color = "rgba(176,176,176,0.5)"  # gray fill
        prod_line_color = "#b0b0b0"           # gray line
        gridcolor = "#eee"
        axis_label_color = "#000"
        axis_tickfont = dict(color="#000", size=13, family="Arial")
        axis_titlefont = dict(color="#000", size=16, family="Arial")

    data = []
    for i, col in enumerate(numeric_cols):
        x_ref = prod_df[col].dropna()
        x_new = new_df[col].dropna()
        xs = np.linspace(min(x_ref.min(), x_new.min()), max(x_ref.max(), x_new.max()), 200)

        kde_ref = gaussian_kde(x_ref)
        kde_new = gaussian_kde(x_new)
        y_ref = kde_ref(xs)
        y_new = kde_new(xs)

        # Reference (gray)
        data.append(go.Scatter(
            x=xs, y=y_ref,
            name="Production Data",
            line=dict(color=prod_line_color, width=2),
            fill='tozeroy',
            visible=(i == 0),
            mode='lines',
            fillcolor=prod_color
        ))
        # New (red)
        data.append(go.Scatter(
            x=xs, y=y_new,
            name="Incoming Data",
            line=dict(color=new_color, width=3),
            visible=(i == 0),
            mode='lines',
            fill='tozeroy',
            fillcolor=new_fill_color
        ))

    steps = []
    for i, col in enumerate(numeric_cols):
        vis = [False]*2*len(numeric_cols)
        vis[2*i] = True
        vis[2*i+1] = True
        steps.append(
            dict(
                label=str(col).capitalize(),
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": {
                            "text": f"Distribution of {str(col).capitalize()}",
                            "x": 0.5,
                            "xanchor": "center"
                        },
                        "xaxis.title": dict(text=str(col).capitalize(), font=axis_titlefont),
                        "xaxis.color": axis_label_color,
                        "xaxis.tickfont": axis_tickfont,
                        "yaxis.title": dict(text="Density", font=axis_titlefont),
                        "yaxis.color": axis_label_color,
                        "yaxis.tickfont": axis_tickfont
                    }
                ]
            )
        )

    layout = go.Layout(
        title={
            "text": f"Distribution of {str(numeric_cols[0]).capitalize()}",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis=dict(
            title=dict(text=str(numeric_cols[0]).capitalize(), font=axis_titlefont),
            gridcolor=gridcolor,
            color=axis_label_color,
            tickfont=axis_tickfont
        ),
        yaxis=dict(
            title=dict(text="Density", font=axis_titlefont),
            gridcolor=gridcolor,
            color=axis_label_color,
            tickfont=axis_tickfont
        ),
        plot_bgcolor=bgcolor,
        paper_bgcolor=bgcolor,
        font=dict(color=textcolor),
        legend=dict(font=dict(color=textcolor)),
        updatemenus=[
            dict(
                buttons=steps,
                direction="down",
                x=0.01, xanchor="left",
                y=1.15, yanchor="top",
                showactive=True,
                bgcolor=bgcolor,
                bordercolor=textcolor,
                font=dict(color=textcolor)
            ),
        ],
        margin=dict(l=40, r=20, t=60, b=40),
        height=height
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def get_plotly_boxplot(
    *,
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    num_summary: pd.DataFrame,   # ← the summary returned by check_numeric_stats
    mode: str = "dark",
    height: int = 300,
):
    """
    Interactive side-by-side box-plots for every numeric feature that shows
    “large” deltas across all summary metrics.

    Parameters
    ----------
    prod_df, new_df : pd.DataFrame
        DataFrames that contain the numeric columns listed in `num_summary`.
    num_summary : pd.DataFrame
        The DataFrame returned by `check_numeric_stats()`.
    mode : {"dark", "light"}
        Colour template toggle.
    height : int
        Base height of the figure in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """

    thresh = 0.5
    metric_cols = ["Mean Δ", "Median Δ", "Std Δ", "Min Δ", "Max Δ"]
    filtered_features = []

    for _, row in num_summary.iterrows():
        all_large = True
        for metric in metric_cols:
            val = row[metric]
            if isinstance(val, str) and val not in ["—", "None"]:
                try:
                    val_num = float(val.replace("%", "").replace("+", "").replace("−", "-").replace("—", "0")) / 100
                except Exception:
                    all_large = False
                    break
                if abs(val_num) < thresh:
                    all_large = False
                    break
            else:
                all_large = False
                break
        if all_large:
            filtered_features.append(row["Feature"])

    if not filtered_features:
        filtered_features = ["None"]

    if mode == "dark":
        bgcolor = "#000"
        textcolor = "#fff"
        prod_color = "#1f77b4"
        new_color = "#ff7f0e"
        gridcolor = "#444"
    else:
        bgcolor = "#fff"
        textcolor = "#000"
        prod_color = "#1f77b4"
        new_color = "#ff7f0e"
        gridcolor = "#eee"

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Production Data", "Incoming Data"),
        horizontal_spacing=0.15
    )

    if filtered_features == ["None"]:
        fig.layout.annotations = []
        for axis in fig.layout:
            if axis.startswith('xaxis') or axis.startswith('yaxis'):
                fig.update_layout({axis: dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title_text="",
                    visible=False,
                )})
        fig.update_layout(
            title=None,
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor,
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        fig.add_annotation(
            text="<b>No Significant Drift Detected<br>in Numeric Features</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=28, color=textcolor),
            align="center"
        )
        return fig

    for i, col in enumerate(filtered_features):
        fig.add_trace(
            go.Box(
                y=prod_df[col],
                name="Production",
                marker=dict(color=prod_color),
                boxpoints="outliers",
                showlegend=False,
                visible=(i == 0)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(
                y=new_df[col],
                name="Incoming",
                marker=dict(color=new_color),
                boxpoints="outliers",
                showlegend=False,
                visible=(i == 0)
            ),
            row=1, col=2
        )

    steps = []
    for i, col in enumerate(filtered_features):
        vis = [False] * (2 * len(filtered_features))
        vis[2 * i] = True
        vis[2 * i + 1] = True
        steps.append(dict(
            label=str(col).capitalize(),
            method="update",
            args=[
                {"visible": vis},
                {"title.text": f"Distribution Comparison: {col.capitalize()}",
                "yaxis.title.text": col.capitalize(),
                "yaxis2.title.text": col.capitalize(),
                "xaxis.gridcolor": gridcolor,
                "xaxis2.gridcolor": gridcolor,
                "yaxis.gridcolor": gridcolor,
                "yaxis2.gridcolor": gridcolor,
                }
            ]
        ))

    fig.update_layout(
        title={"text": f"Distribution Comparison: {str(filtered_features[0]).capitalize()}",
            "x": 0.5, "xanchor": "center"},
        plot_bgcolor=bgcolor, paper_bgcolor=bgcolor, font=dict(color=textcolor),
        updatemenus=[
            dict(
                buttons=steps,
                direction="down",
                x=0.01, xanchor="left",
                y=1.15, yanchor="top",
                showactive=True,
                bgcolor=bgcolor,
                bordercolor=textcolor,
                font=dict(color=textcolor)
            ),
        ],
        margin=dict(l=40, r=20, t=60, b=40),
        height=height,

    )
    fig.update_xaxes(title_text="", gridcolor=gridcolor, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="", gridcolor=gridcolor, showticklabels=False, row=1, col=2)
    fig.update_yaxes(
        title_text=str(filtered_features[0]).capitalize(),
        gridcolor=gridcolor,
        color="black",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text=str(filtered_features[0]).capitalize(),
        gridcolor=gridcolor,
        color="black",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        row=1, col=2
    )

    return fig

# ---------------------------------------------------------------------------
# Numeric drift check function that returns a NumStatsSummary
# ---------------------------------------------------------------------------
def check_numeric_stats(
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    num_cols: list[str],
    p_threshold: float = 0.05,
    ks_alternative: str = "two-sided",
) -> NumStatsSummary:
    """
    Compare numeric columns and flag drift using Levene (variance) and
    Kolmogorov–Smirnov (shape) tests.  Returns a NumStatsSummary that
    supports .interpret().
    """
    def pct_change(a, b):
        if pd.isna(a) or a == 0 or pd.isna(b):
            return "—"
        return f"{(b - a) / abs(a):+0.2%}"

    corrected_alpha = p_threshold / len(num_cols)
    rows = []

    for col in num_cols:
        ref, new = prod_df[col], new_df[col]

        stats_ref = {
            "mean":   ref.mean(),
            "median": ref.median(),
            "std":    ref.std(ddof=1),
            "min":    ref.min(),
            "max":    ref.max(),
            "nulls":  ref.isna().mean(),
        }
        stats_new = {
            "mean":   new.mean(),
            "median": new.median(),
            "std":    new.std(ddof=1),
            "min":    new.min(),
            "max":    new.max(),
            "nulls":  new.isna().mean(),
        }
        deltas = {k: pct_change(stats_ref[k], stats_new[k]) for k in stats_ref}

        _, p_levene = stats.levene(ref, new, center="median")
        _, p_ks     = stats.ks_2samp(
            ref, new, alternative=ks_alternative, mode="asymp"
        )

        if (p_levene <= p_threshold) and (p_ks <= corrected_alpha):
            flag = "🔴"
        elif (p_levene <= p_threshold) or (p_ks <= corrected_alpha):
            flag = "🟡"
        else:
            flag = "🟢"

        rows.append({
            "Feature":  col,
            "Mean Δ":   deltas["mean"],
            "Median Δ": deltas["median"],
            "Std Δ":    deltas["std"],
            "Min Δ":    deltas["min"],
            "Max Δ":    deltas["max"],
            #"Nulls Δ":  deltas["nulls"],
            "Levene p": f"{p_levene:.3g}",
            "KS p":     f"{p_ks:.3g}",
            "Detection": flag,
        })

    df = NumStatsSummary(rows)
    df._p_threshold      = p_threshold
    df._corrected_alpha  = corrected_alpha
    df._ks_alternative   = ks_alternative
    return df



######################### CONCEPT DRIFT ############################

def check_numeric_correlation(
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    num_cols: list[str],
    target: str,
    target_type: str = "auto",  # "auto", "continuous", or "categorical"
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    cat_max_uniques: int = 2
) -> pd.DataFrame:
    """
    Compares drift in relationship between numeric features and the target variable (continuous or categorical)
    in both reference and new DataFrames.

    For continuous targets: Uses Pearson correlation.
    For categorical targets: Uses ANOVA F-value.

    Returns the *signed* change (not absolute).
    """
    import warnings
    from pandas.api.types import CategoricalDtype
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    results = []
    if target_type == "auto":
        nunique = prod_df[target].nunique()
        if prod_df[target].dtype == 'O' or nunique <= cat_max_uniques:
            use_mode = "categorical"
        else:
            use_mode = "continuous"
    else:
        use_mode = target_type.lower()

    for col in num_cols:
        if col not in prod_df.columns or col not in new_df.columns:
            continue

        flag = None
        if use_mode == "continuous":
            # Pearson correlation
            ref_valid = prod_df[[col, target]].dropna()
            new_valid = new_df[[col, target]].dropna()
            if ref_valid.nunique().min() < 2 or new_valid.nunique().min() < 2:
                ref_stat = np.nan
                new_stat = np.nan
                change = np.nan
                flag = "N/A"
            else:
                ref_stat = ref_valid.corr().iloc[0, 1]
                new_stat = new_valid.corr().iloc[0, 1]
                change = new_stat - ref_stat  # *** signed difference ***
        elif use_mode == "categorical":
            try:
                ref_valid = prod_df[[col, target]].dropna()
                new_valid = new_df[[col, target]].dropna()
                from pandas.api.types import CategoricalDtype
                if not isinstance(ref_valid[target].dtype, CategoricalDtype):
                    ref_valid[target] = ref_valid[target].astype('category')
                if not isinstance(new_valid[target].dtype, CategoricalDtype):
                    new_valid[target] = new_valid[target].astype('category')
                if ref_valid[target].nunique() < 2 or new_valid[target].nunique() < 2:
                    ref_stat, new_stat, change, flag = np.nan, np.nan, np.nan, "N/A"
                else:
                    def anova_f(X, y):
                        groups = [X[y == cat] for cat in y.cat.categories]
                        means = [g.mean() for g in groups]
                        n = [len(g) for g in groups]
                        overall_mean = X.mean()
                        SSB = sum([ni * (mi - overall_mean) ** 2 for ni, mi in zip(n, means)])
                        SSW = sum([((g - mi) ** 2).sum() for g, mi in zip(groups, means)])
                        dfb = len(groups) - 1
                        dfw = X.shape[0] - len(groups)
                        MSB = SSB / dfb if dfb > 0 else np.nan
                        MSW = SSW / dfw if dfw > 0 else np.nan
                        F = MSB / MSW if (MSB is not np.nan and MSW not in [0, np.nan]) else np.nan
                        return F
                    ref_stat = anova_f(ref_valid[col], ref_valid[target])
                    new_stat = anova_f(new_valid[col], new_valid[target])
                    if ref_stat is not None and not np.isnan(ref_stat) and ref_stat != 0:
                        change = (new_stat - ref_stat) / abs(ref_stat)  # *** signed, relative change ***
                    else:
                        change = new_stat - ref_stat  # *** signed, not absolute ***
            except Exception as e:
                ref_stat, new_stat, change, flag = np.nan, np.nan, np.nan, "N/A"
        else:
            raise ValueError("target_type must be 'continuous', 'categorical', or 'auto'.")

        if flag is None:
            if pd.isna(change):
                flag = "N/A"
            elif abs(change) >= high_threshold:
                flag = "🔴"
            elif abs(change) >= low_threshold:
                flag = "🟡"
            else:
                flag = "🟢"

        results.append({
            'Feature': col,
            'Reference_stat': f"{ref_stat:+.3f}",
            'New_stat': f"{new_stat:+.3f}",
            'Change': f"{change:+.3f}",     # *** now signed ***
            'Detection': flag,
            #'Mode': use_mode
        })

    return pd.DataFrame(results)


