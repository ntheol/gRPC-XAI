import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

def plot_1D_continuous_eff(res_df, X, fig=None, ax=None):
    """Plot the 1D ALE plot for a continuous feature.

    Arguments:
    res_df -- A pandas DataFrame containing the computed effects
    (the output of ale_1D_continuous).
    X -- The dataset used to compute the effects.
    fig, ax -- matplotlib figure and axis.
    """

    feature_name = res_df.index.name
    # position: jitter
    # to see the distribution of the data points clearer, each point x will be nudged a random value between
    # -0.5*(distance from the bin's lower value) and +0.5*(distance from bin's upper value)
    jitter_limits = pd.DataFrame(
        {
            "x": X[feature_name],
            "bin_code": pd.cut(
                X[feature_name], res_df.index.to_list(), include_lowest=True
            ).cat.codes
            + 1,
        }
    ).assign(
        jitter_step_min=lambda df: (df["x"] - res_df.index[df["bin_code"] - 1]) * 0.5,
        jitter_step_max=lambda df: (res_df.index[df["bin_code"]] - df["x"]) * 0.5,
    )
    np.random.seed(123)
    rug = jitter_limits.apply(
        lambda row: row["x"]
        + np.random.uniform(-row["jitter_step_min"], row["jitter_step_max"]),
        axis=1,
    )

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res_df[["eff"]])
    tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-5, units="points")
    ax.plot(
        rug,
        [res_df.drop("size", axis=1).min().min()] * len(rug),
        "|",
        color="k",
        alpha=0.2,
        transform=tr,
    )
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        label = lowerCI_name.str.split("_")[0][1] + " confidence interval"
        ax.fill_between(
            res_df.index,
            y1=res_df[lowerCI_name[0]],
            y2=res_df[upperCI_name[0]],
            alpha=0.2,
            color="grey",
            label=label,
        )
        ax.legend()
    ax.set_xlabel(res_df.index.name)
    ax.set_ylabel("Effect on prediction (centered)")
    ax.set_title("1D ALE Plot - Continuous")
    return fig, ax


def plot_1D_discrete_categorical_eff(res_df, X, fig=None, ax=None):
    """Plot the 1D ALE plot for a discrete feature.

    Arguments:
    res_df -- A pandas DataFrame with the computed effects
    (the output of ale_1D_discrete).
    X -- The dataset used to compute the effects.
    fig, ax -- matplotlib figure and axis.
    """

    feature_name = res_df.index.name
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Effect on prediction (centered)")
    yerr = 0
    lowerCI_name = res_df.columns[res_df.columns.str.contains("lowerCI")]
    upperCI_name = res_df.columns[res_df.columns.str.contains("upperCI")]
    if (len(lowerCI_name) == 1) and (len(upperCI_name) == 1):
        yerr = res_df[upperCI_name].subtract(res_df["eff"], axis=0).iloc[:, 0]
    ax.errorbar(
        res_df.index.astype(str),
        res_df["eff"],
        yerr=yerr,
        capsize=3,
        marker="o",
        linestyle="dashed",
        color="black",
    )
    ax2 = ax.twinx()
    ax2.set_ylabel("Size", color="lightblue")
    ax2.bar(res_df.index.astype(str), res_df["size"], alpha=0.1, align="center")
    ax2.tick_params(axis="y", labelcolor="lightblue")
    ax2.set_title("1D ALE Plot - Discrete/Categorical")
    fig.tight_layout()
    return fig, ax, ax2