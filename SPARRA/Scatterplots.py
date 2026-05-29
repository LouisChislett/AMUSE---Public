"""
Scatterplots.py - SPARRA visualisations in the same style as the electricity dataset plots.

This script expects the CSV files written by the updated SPARRA Results.py, namely:
  - Outputs/num_updates_cost_005.csv
  - Outputs/num_updates_cost_04.csv
  - Outputs/average_auc_cost_005.csv OR Outputs/average_auc_score_cost_005.csv
  - Outputs/average_auc_cost_04.csv OR Outputs/average_auc_score_cost_04.csv
  - Outputs/utilities.csv

It produces, for each cost penalty:
  - Average AUC vs Number of Updates
  - Cumulative Utility vs Number of Updates

As in the electricity plotting script, Always Update and Never Update are excluded.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

# Deliberately do not use adjustText here. With very tight x/y ranges,
# adjustText can move labels far outside the axes and create huge blank plots.
HAS_ADJUST_TEXT = False

# ============================================================
# Paths
# ============================================================

OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

UTILITIES_PATH = os.path.join(OUTPUTS_DIR, "utilities.csv")

COSTS = [0.05, 0.4]
METHODS = ["AMUSE", "DDM", "FRD", "STEPD", "Random", "Even"]

# Results.py has had two naming variants while iterating. Support both.
AUC_FILE_CANDIDATES = {
    0.05: [
        os.path.join(OUTPUTS_DIR, "average_auc_cost_005.csv"),
        os.path.join(OUTPUTS_DIR, "average_auc_score_cost_005.csv"),
    ],
    0.4: [
        os.path.join(OUTPUTS_DIR, "average_auc_cost_04.csv"),
        os.path.join(OUTPUTS_DIR, "average_auc_score_cost_04.csv"),
    ],
}

UPDATES_FILE = {
    0.05: os.path.join(OUTPUTS_DIR, "num_updates_cost_005.csv"),
    0.4: os.path.join(OUTPUTS_DIR, "num_updates_cost_04.csv"),
}

# ============================================================
# Loading / standardising helpers
# ============================================================


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find any of these expected files:\n" + "\n".join(paths)
    )


def standardise_two_column_csv(path, value_column):
    """Read a two-column CSV whose first column is the method index/name."""
    df = pd.read_csv(path)
    df = df.iloc[:, :2].copy()
    df.columns = ["Method", value_column]
    df["Method"] = df["Method"].astype(str).str.strip()
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    return df.dropna(subset=[value_column])


def load_num_updates(cost):
    return standardise_two_column_csv(UPDATES_FILE[cost], "Number of Updates")


def load_average_auc(cost):
    auc_path = first_existing_path(AUC_FILE_CANDIDATES[cost])
    return standardise_two_column_csv(auc_path, "Average AUC")


def load_utilities():
    utilities = pd.read_csv(UTILITIES_PATH)
    utilities = utilities.rename(columns={utilities.columns[0]: "Method"})
    utilities["Method"] = utilities["Method"].astype(str).str.strip()
    utilities["Method"] = utilities["Method"].replace(
        {
            "RL": "AMUSE",
            "FRD": "FRD",
            "Always": "Always Update",
            "Never": "Never Update",
        }
    )

    # Normalise cost column names so either 0.4 or 0.40 style inputs work.
    renamed_cost_cols = {}
    for col in utilities.columns:
        if col == "Method":
            continue
        try:
            renamed_cost_cols[col] = str(float(col))
        except ValueError:
            renamed_cost_cols[col] = col
    utilities = utilities.rename(columns=renamed_cost_cols)

    for col in utilities.columns:
        if col != "Method":
            utilities[col] = pd.to_numeric(utilities[col], errors="coerce")

    return utilities


utilities = load_utilities()

# ============================================================
# Helper functions
# ============================================================


def get_value(df, method, column):
    match = df.loc[df["Method"] == method, column]

    if match.empty:
        raise KeyError(
            f"Could not find method '{method}' in {column}.\n"
            f"Available methods are:\n{df['Method'].tolist()}"
        )

    value = match.iloc[0]
    if pd.isna(value):
        raise ValueError(f"Value for method '{method}' in column '{column}' is NaN.")

    return float(value)


def make_auc_data(cost):
    average_auc = load_average_auc(cost)
    num_updates = load_num_updates(cost)

    rows = []
    for method in METHODS:
        rows.append(
            {
                "Method": method,
                "Number of Updates": get_value(
                    num_updates,
                    method,
                    "Number of Updates",
                ),
                "Average AUC": get_value(
                    average_auc,
                    method,
                    "Average AUC",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_utility_data(cost):
    num_updates = load_num_updates(cost)
    cost_text = str(cost)

    if cost_text not in utilities.columns:
        raise KeyError(
            f"Could not find cost column '{cost_text}' in utilities.csv.\n"
            f"Available columns are:\n{utilities.columns.tolist()}"
        )

    rows = []
    for method in METHODS:
        rows.append(
            {
                "Method": method,
                "Number of Updates": get_value(
                    num_updates,
                    method,
                    "Number of Updates",
                ),
                "Utility": get_value(
                    utilities,
                    method,
                    cost_text,
                ),
            }
        )

    return pd.DataFrame(rows)


def label_offsets_for_overlaps(df, y_col):
    """Return deterministic staggered label offsets for labels that would collide."""
    offsets = {}

    # Treat visually identical points as overlaps. Rounding avoids tiny floating
    # point differences preventing labels from being staggered.
    grouped = df.assign(
        _x_key=df["Number of Updates"].round(8),
        _y_key=df[y_col].round(8),
    ).groupby(["_x_key", "_y_key"], sort=False)

    # Offsets are in screen points, not data units, so they stay sensible even
    # when the x-axis spans only 0--3 updates.
    exact_overlap_offsets = [
        (8, 8),
        (8, -16),
        (-52, 8),
        (-52, -16),
        (8, 28),
        (-52, 28),
        (8, -36),
        (-52, -36),
    ]

    for _, group in grouped:
        if len(group) == 1:
            offsets[group.index[0]] = (8, 8)
        else:
            for j, idx in enumerate(group.index):
                offsets[idx] = exact_overlap_offsets[j % len(exact_overlap_offsets)]

    # Also stagger labels that have the same x value and very similar y values.
    # This handles points that are not mathematically identical but still overlap
    # visually because the AUC range is tiny.
    y_range = df[y_col].max() - df[y_col].min()
    if y_range == 0:
        y_range = 1.0
    close_y_threshold = 0.03 * y_range

    for _, x_group in df.groupby(df["Number of Updates"].round(8), sort=False):
        ordered = x_group.sort_values(y_col)
        close_cluster = []

        def flush_cluster(cluster):
            if len(cluster) <= 1:
                return
            cluster_offsets = [
                (8, 16),
                (8, -20),
                (-60, 16),
                (-60, -20),
                (8, 34),
                (-60, 34),
            ]
            for j, idx in enumerate(cluster):
                offsets[idx] = cluster_offsets[j % len(cluster_offsets)]

        previous_y = None
        for idx, row in ordered.iterrows():
            current_y = row[y_col]
            if previous_y is None or abs(current_y - previous_y) <= close_y_threshold:
                close_cluster.append(idx)
            else:
                flush_cluster(close_cluster)
                close_cluster = [idx]
            previous_y = current_y
        flush_cluster(close_cluster)

    return offsets

def plot_scatter(df, y_col, title, filename):
    """Same simple scatter style as the electricity script, with non-overlapping labels."""
    plt.figure(figsize=(10, 6))

    plt.scatter(df["Number of Updates"], df[y_col], s=80)

    label_offsets = label_offsets_for_overlaps(df, y_col)
    texts = []
    for idx, row in df.iterrows():
        text = plt.annotate(
            row["Method"],
            (row["Number of Updates"], row[y_col]),
            textcoords="offset points",
            xytext=label_offsets[idx],
            fontsize=9,
        )
        texts.append(text)

    plt.xlabel("Number of Updates", fontsize=18, fontweight="bold")
    plt.ylabel(y_col, fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Use tight, data-driven x-axis padding. Allow x_min below 0 so points at
    # exactly 0 updates are not clipped by the left axis.
    x_values = df["Number of Updates"]
    x_range = x_values.max() - x_values.min()
    x_pad = max(0.30, 0.08 * x_range)
    x_min = x_values.min() - x_pad
    x_max = x_values.max() + x_pad

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    y_min = df[y_col].min() - 0.01 * abs(df[y_col].min())
    y_max = df[y_col].max() + 0.01 * abs(df[y_col].max())

    if y_col == "Average AUC":
        y_min = max(0, y_min)
        y_max = min(1, y_max)

    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Keep argument for parity with electricity script, but do not display a title
    # because the attached script accepted a title but did not call plt.title().
    _ = title

    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


# ============================================================
# Generate plots
# ============================================================

for cost in COSTS:
    cost_label = str(cost).replace(".", "_")

    auc_df = make_auc_data(cost)
    auc_data_path = os.path.join(OUTPUTS_DIR, f"scatterplot_auc_data_{cost_label}.csv")
    auc_df.to_csv(auc_data_path, index=False)

    plot_scatter(
        auc_df,
        y_col="Average AUC",
        title=f"Average AUC vs Number of Updates, Cost = {cost}",
        filename=f"average_auc_vs_updates_cost_{cost_label}.png",
    )

    utility_df = make_utility_data(cost)
    utility_data_path = os.path.join(OUTPUTS_DIR, f"scatterplot_utility_data_{cost_label}.csv")
    utility_df.to_csv(utility_data_path, index=False)

    plot_scatter(
        utility_df,
        y_col="Utility",
        title=f"Cumulative Utility vs Number of Updates, Cost = {cost}",
        filename=f"utility_vs_updates_cost_{cost_label}.png",
    )

print("Done.")
