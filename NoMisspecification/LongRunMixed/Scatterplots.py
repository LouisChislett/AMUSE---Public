import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

# ============================================================
# Paths
# ============================================================

OUTPUTS_DIR = "Outputs"

average_updates_path = os.path.join(OUTPUTS_DIR, "average_updates.csv")
average_accuracies_path = os.path.join(OUTPUTS_DIR, "average_accuracies.csv")
utilities_path = os.path.join(OUTPUTS_DIR, "utilities.csv")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ============================================================
# Load CSV files
# ============================================================

average_updates = pd.read_csv(average_updates_path)
average_accuracies = pd.read_csv(average_accuracies_path)
utilities = pd.read_csv(utilities_path)

# Standardise average_updates.csv
average_updates = average_updates.iloc[:, :2].copy()
average_updates.columns = ["Method", "Number of Updates"]
average_updates["Method"] = average_updates["Method"].astype(str).str.strip()
average_updates["Number of Updates"] = pd.to_numeric(
    average_updates["Number of Updates"], errors="coerce"
)

# Standardise average_accuracies.csv
average_accuracies = average_accuracies.iloc[:, :2].copy()
average_accuracies.columns = ["Method", "Average Accuracy"]
average_accuracies["Method"] = average_accuracies["Method"].astype(str).str.strip()
average_accuracies["Average Accuracy"] = pd.to_numeric(
    average_accuracies["Average Accuracy"], errors="coerce"
)

# Standardise utilities.csv
utilities = utilities.rename(columns={utilities.columns[0]: "Method"})
utilities["Method"] = utilities["Method"].astype(str).str.strip()
utilities["Method"] = utilities["Method"].replace(
    {
        "RL": "AMUSE",
        "RL (Transfer)": "AMUSE Transfer",
        "Always": "Always Update",
        "Never": "Never Update",
    }
)

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

    return float(match.iloc[0])


def make_accuracy_data(cost):
    cost_text = str(cost)

    # Always Update and Never Update removed.
    # AMUSE (static) is regular AMUSE; AMUSE (dynamic) is AMUSE Transfer.
    methods = [
        ("AMUSE (static)", f"AMUSE ({cost_text})", f"AMUSE ({cost_text})"),
        ("AMUSE (dynamic)", f"AMUSE Transfer ({cost_text})", f"AMUSE Transfer ({cost_text})"),
        ("DDM", "DDM", "DDM"),
        ("FRD", "FRD", "FRD"),
        ("STEPD", "STEPD", "STEPD"),
        ("Random", f"Random ({cost_text})", f"Random ({cost_text})"),
        ("Even", f"Even ({cost_text})", f"Even ({cost_text})"),
    ]

    rows = []

    for display_name, updates_name, accuracy_name in methods:
        rows.append(
            {
                "Method": display_name,
                "Number of Updates": get_value(
                    average_updates,
                    updates_name,
                    "Number of Updates",
                ),
                "Average Accuracy": get_value(
                    average_accuracies,
                    accuracy_name,
                    "Average Accuracy",
                ),
            }
        )

    return pd.DataFrame(rows)


def make_utility_data(cost):
    cost_text = str(cost)

    # Always Update and Never Update removed.
    # AMUSE (static) is regular AMUSE; AMUSE (dynamic) is AMUSE Transfer.
    methods = [
        ("AMUSE (static)", f"AMUSE ({cost_text})", "AMUSE"),
        ("AMUSE (dynamic)", f"AMUSE Transfer ({cost_text})", "AMUSE Transfer"),
        ("DDM", "DDM", "DDM"),
        ("FRD", "FRD", "FRD"),
        ("STEPD", "STEPD", "STEPD"),
        ("Random", f"Random ({cost_text})", "Random"),
        ("Even", f"Even ({cost_text})", "Even"),
    ]

    rows = []

    for display_name, updates_name, utility_name in methods:
        rows.append(
            {
                "Method": display_name,
                "Number of Updates": get_value(
                    average_updates,
                    updates_name,
                    "Number of Updates",
                ),
                "Utility": get_value(
                    utilities,
                    utility_name,
                    cost_text,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_scatter(df, y_col, filename):
    plt.figure(figsize=(10, 6))

    plt.scatter(df["Number of Updates"], df[y_col], s=80)

    x_values = df["Number of Updates"]
    y_values = df[y_col]

    # Use range-based padding rather than a fixed padding of 5.
    x_range = x_values.max() - x_values.min()
    y_range = y_values.max() - y_values.min()

    # Handle the unlikely case where all values are identical.
    if x_range == 0:
        x_range = max(1, abs(x_values.max()))
    if y_range == 0:
        y_range = max(0.01, abs(y_values.max()))

    x_padding = 0.08 * x_range
    y_padding = 0.08 * y_range

    x_min = max(0, x_values.min() - x_padding)
    x_max = x_values.max() + x_padding
    y_min = y_values.min() - y_padding
    y_max = y_values.max() + y_padding

    if y_col == "Average Accuracy":
        y_min = max(0, y_min)
        y_max = min(1, y_max)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Annotate points. Move labels inward for points near the right edge.
    for _, row in df.iterrows():
        x = row["Number of Updates"]
        y = row[y_col]

        if x > x_values.min() + 0.85 * x_range:
            xytext = (-10, 6)
            ha = "right"
        else:
            xytext = (6, 6)
            ha = "left"

        plt.annotate(
            row["Method"],
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=9,
            clip_on=False,
        )

    plt.xlabel("Number of Updates", fontsize=18, fontweight="bold")
    plt.ylabel(y_col, fontsize=18, fontweight="bold")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    print(f"Saved: {save_path}")

# ============================================================
# Generate plots
# ============================================================

for cost in [0.05, 0.4]:
    cost_label = str(cost).replace(".", "_")

    accuracy_df = make_accuracy_data(cost)
    accuracy_df.to_csv(
        os.path.join(OUTPUTS_DIR, f"scatterplot_accuracy_data_{cost_label}.csv"),
        index=False,
    )

    plot_scatter(
        accuracy_df,
        y_col="Average Accuracy",
        filename=f"average_accuracy_vs_updates_cost_{cost_label}.png",
    )

    utility_df = make_utility_data(cost)
    utility_df.to_csv(
        os.path.join(OUTPUTS_DIR, f"scatterplot_utility_data_{cost_label}.csv"),
        index=False,
    )

    plot_scatter(
        utility_df,
        y_col="Utility",
        filename=f"utility_vs_updates_cost_{cost_label}.png",
    )

print("Done.")
