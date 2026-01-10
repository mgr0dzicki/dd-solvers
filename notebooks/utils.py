import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.titlesize": 14,
        "font.size": 12,
    }
)

MEASURED_BANDWIDTH = 1774
MEASURED_BANDWIDTH_STR = "1,774GB/s"

def extract_row_metadata(row):
    metadata = (
        ast.literal_eval(row["metadata"])
        if "metadata" in row and not pd.isna(row["metadata"])
        else {}
    )
    setup_metadata = (
        ast.literal_eval(row["setup metadata"])
        if "setup metadata" in row and not pd.isna(row["setup metadata"])
        else {}
    )

    return pd.Series(
        {
            "iterations": metadata.get("iterations"),
            "asm time": row.get("solve time") if "CG" not in row["solver"] else None,
            "cg time": row.get("solve time") if "CG" in row["solver"] else None,
            "local solvers time": setup_metadata.get("local solver solve time"),
            "coarse solver time": setup_metadata.get("coarse solver solve time"),
            "local solvers setup time": setup_metadata.get("local solver setup time"),
            "coarse solver setup time": setup_metadata.get("coarse solver setup time"),
            "asm setup time": (
                row.get("solver setup time") if "CG" not in row["solver"] else None
            ),
            "cg setup time": (
                row.get("solver setup time") if "CG" in row["solver"] else None
            ),
        }
    )


def plot_clustered_stacked(
    ax,
    df_list,
    labels=None,
    cluster_width=0.8,
    hatch_list=None,
    alpha_list=None,
    cmap=lambda x: plt.cm.Set2(1 - x),
    add_legend=True,
):
    if len(df_list) == 0:
        raise ValueError("df_list must contain at least one DataFrame")
    n_df = len(df_list)
    cols = df_list[0].columns
    idx = df_list[0].index
    n_col = len(cols)
    n_ind = len(idx)
    for df in df_list:
        if not df.columns.equals(cols) or not df.index.equals(idx):
            raise ValueError("All dataframes must have identical columns and index")

    if hatch_list is None:
        hatch_list = [None] * n_df
    if alpha_list is None:
        alpha_list = [1.0] * n_df

    x = np.arange(n_ind)
    bar_w = float(cluster_width) / n_df
    offsets = (np.arange(n_df) - (n_df - 1) / 2.0) * bar_w
    color_vals = [cmap(i / max(1, n_col - 1)) for i in range(n_col)]

    for i, df in enumerate(df_list):
        xpos = x + offsets[i]
        bottoms = np.zeros(n_ind)
        for j, col in enumerate(cols):
            heights = df[col].values
            col_label = col if i == 0 else None  # label each component only once
            ax.bar(
                xpos,
                heights,
                width=bar_w,
                bottom=bottoms,
                label=col_label,
                edgecolor="white",
                linewidth=0.5,
                color=color_vals[j],
                hatch=hatch_list[i],
                alpha=alpha_list[i],
            )
            bottoms = bottoms + heights

    ax.set_xticks(x)
    ax.set_xticklabels(idx, rotation=0)
    left = x.min() - cluster_width / 2
    right = x.max() + cluster_width / 2
    ax.set_xlim(left - 0.1, right + 0.1)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # place per-axis legends only if requested
    if add_legend:
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            leg1 = ax.legend(
                handles,
                labels_,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                title="Components",
            )
            ax.add_artist(leg1)
        if labels is not None:
            proxies = []
            for i, lab in enumerate(labels):
                p = mpatches.Patch(
                    facecolor="gray",
                    edgecolor="black",
                    hatch=hatch_list[i] or "",
                    alpha=alpha_list[i],
                    label=lab,
                )
                proxies.append(p)
            ax.legend(
                proxies,
                labels,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                title="Precisions",
            )

    return ax


# def format_mesh(mesh: tuple[int, str]):
#     k, m = mesh
#     return f"\\mathcal{{{m}}}_{{{int(k)}}}"

def format_mesh(mesh: str):
    if len(mesh) == 1:
        return mesh
    m, k = mesh[0], int(mesh[1:])
    return f"\\mathcal{{{m}}}_{{{int(k)}}}"


def safe_map(func):
    def wrapper(x):
        try:
            return func(x)
        except Exception:
            return None

    return wrapper
