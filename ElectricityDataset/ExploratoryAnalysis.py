from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

OUTPUT_DIR = Path("ExploratoryAnalysis")
OUTPUT_DIR.mkdir(exist_ok=True)

OPENML_DATASET_ID = 151
TARGET_COL = "class"

FEATURE_COLS_EXPECTED = [
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]

FEATURE_LABELS = {
    "nswprice": "NSW price",
    "nswdemand": "NSW demand",
    "vicprice": "VIC price",
    "vicdemand": "VIC demand",
    "transfer": "Transfer",
}

ROLLING_CLASS_WINDOWS = [1000]
ROLLING_FEATURE_WINDOW = 1000

FIGSIZE_STANDARD = (8, 5)
FIGSIZE_WIDE = (10, 5)
FIGSIZE_GRID = (12, 8)

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11
SUBPLOT_TITLE_SIZE = 13

AXIS_LABEL_WEIGHT = "bold"

plt.rcParams.update({
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.labelweight": AXIS_LABEL_WEIGHT,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": 120,
})


def save_current_figure(filename):
    """Save the current matplotlib figure neatly and close it."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def style_axis(ax, xlabel=None, ylabel=None):
    """Apply consistent thesis-friendly axis styling."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight=AXIS_LABEL_WEIGHT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontweight=AXIS_LABEL_WEIGHT)

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


# ------------------------------------------------------------
# 2. Load ELEC2 from OpenML
# ------------------------------------------------------------

dataset = openml.datasets.get_dataset(OPENML_DATASET_ID)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe"
)

df = X.copy()
df[TARGET_COL] = y

# Standardise column names and labels
df.columns = [str(c).strip().lower() for c in df.columns]
df[TARGET_COL] = df[TARGET_COL].astype(str).str.upper().str.strip()

# Preserve stream order
df["stream_index"] = np.arange(len(df))

FEATURE_COLS = [col for col in FEATURE_COLS_EXPECTED if col in df.columns]

if not FEATURE_COLS:
    raise ValueError("None of the expected ELEC2 feature columns were found.")

print("Loaded ELEC2 from OpenML")
print(f"Dataset name: {dataset.name}")
print(f"Dataset ID: {OPENML_DATASET_ID}")
print(f"Shape: {df.shape}")
print(f"Features used: {FEATURE_COLS}")
print(f"Target labels: {sorted(df[TARGET_COL].unique())}")


# ------------------------------------------------------------
# 3. Estimate earliest and latest timestamp
# ------------------------------------------------------------

def estimate_elec2_timestamps(elec_df):
    """
    Estimate timestamps for ELEC2 using the available date/period columns.

    Important:
    OpenML's ELEC2 'date' column is numeric/normalised, so we should not
    pass it directly to pd.to_datetime(). If we do, pandas interprets it as
    a Unix timestamp near 1970.

    The ELEC2 benchmark is commonly described as covering the period from
    7 May 1996 to 5 December 1998. We therefore map the numeric date column
    onto this known date range for descriptive reporting.

    The resulting timestamp is only for descriptive EDA. The original stream
    order remains the main ordering used in the analysis.
    """

    out = elec_df.copy()

    known_start = pd.Timestamp("1996-05-07 00:00:00")
    known_end = pd.Timestamp("1998-12-05 23:30:00")

    if "date" not in out.columns:
        out["estimated_timestamp"] = (
            known_start + pd.to_timedelta(out["stream_index"] * 30, unit="m")
        )
        return out

    date_numeric = pd.to_numeric(out["date"], errors="coerce")

    if date_numeric.notna().mean() > 0.8:
        date_min = date_numeric.min()
        date_max = date_numeric.max()

        if date_max == date_min:
            out["estimated_timestamp"] = known_start
            return out

        # Map normalised date values onto the known ELEC2 date range
        date_scaled = (date_numeric - date_min) / (date_max - date_min)

        out["estimated_timestamp"] = (
            known_start + date_scaled * (known_end - known_start)
        )

        return out

    # Only try direct parsing if the date column is not numeric
    parsed_dates = pd.to_datetime(out["date"], errors="coerce")

    if parsed_dates.notna().mean() > 0.8:
        out["estimated_timestamp"] = parsed_dates
        return out

    # Final fallback: assume stream order at half-hourly frequency
    out["estimated_timestamp"] = (
        known_start + pd.to_timedelta(out["stream_index"] * 30, unit="m")
    )

    return out


df = estimate_elec2_timestamps(df)

earliest_timestamp = df["estimated_timestamp"].min()
latest_timestamp = df["estimated_timestamp"].max()

print(f"Earliest estimated timestamp: {earliest_timestamp}")
print(f"Latest estimated timestamp:   {latest_timestamp}")


# ------------------------------------------------------------
# 4. Encode target variable
# ------------------------------------------------------------

positive_labels = {"UP", "TRUE", "1", "YES"}

df["class_binary"] = df[TARGET_COL].apply(
    lambda value: 1 if str(value).upper().strip() in positive_labels else 0
)

# Safety check: make sure target is binary
if df["class_binary"].nunique() != 2:
    raise ValueError(
        "Target encoding did not produce two classes. "
    )


# ------------------------------------------------------------
# 5. Save tables
# ------------------------------------------------------------

class_counts = df[TARGET_COL].value_counts()
class_proportions = df[TARGET_COL].value_counts(normalize=True)

chapter_dataset_summary = pd.DataFrame([
    {
        "quantity": "Source",
        "value": f"OpenML dataset {OPENML_DATASET_ID} ({dataset.name})",
    },
    {
        "quantity": "Number of observations",
        "value": len(df),
    },
    {
        "quantity": "Number of predictor variables used",
        "value": len(FEATURE_COLS),
    },
    {
        "quantity": "Predictor variables used",
        "value": ", ".join(FEATURE_COLS),
    },
    {
        "quantity": "Target variable",
        "value": TARGET_COL,
    },
    {
        "quantity": "Class labels",
        "value": ", ".join(sorted(df[TARGET_COL].unique())),
    },
    {
        "quantity": "Task",
        "value": "Binary classification",
    },
    {
        "quantity": "Ordering",
        "value": "Original stream order",
    },
    {
        "quantity": "Earliest estimated timestamp",
        "value": str(earliest_timestamp),
    },
    {
        "quantity": "Latest estimated timestamp",
        "value": str(latest_timestamp),
    },
    {
        "quantity": "Missing values in selected variables",
        "value": int(df[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()),
    },
])

chapter_dataset_summary.to_csv(
    OUTPUT_DIR / "chapter_dataset_summary.csv",
    index=False
)

chapter_class_balance = pd.DataFrame({
    "class": class_counts.index,
    "count": class_counts.values,
    "proportion": class_proportions.loc[class_counts.index].values,
})

chapter_class_balance.to_csv(
    OUTPUT_DIR / "chapter_class_balance.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Class distribution figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

ax.bar(
    chapter_class_balance["class"],
    chapter_class_balance["count"],
)

style_axis(
    ax,
    xlabel="Class",
    ylabel="Number of observations",
)

save_current_figure("fig_class_distribution.png")


# ------------------------------------------------------------
# 7. Rolling class proportion figure
# ------------------------------------------------------------

for window in ROLLING_CLASS_WINDOWS:
    df[f"rolling_up_{window}"] = (
        df["class_binary"]
        .rolling(window, min_periods=max(1, window // 5))
        .mean()
    )

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

for window in ROLLING_CLASS_WINDOWS:
    ax.plot(
        df["stream_index"],
        df[f"rolling_up_{window}"],
        label=f"Rolling mean proportion",
        linewidth=2,
    )

ax.axhline(
    df["class_binary"].mean(),
    linestyle="--",
    linewidth=2,
    label="Overall UP proportion",
)

style_axis(
    ax,
    xlabel="Stream index",
    ylabel="Proportion UP",
)

ax.legend(frameon=False)

save_current_figure("fig_rolling_class_proportion.png")


# ------------------------------------------------------------
# 8. mean explanatory variables by half-hour trading period figures
# ------------------------------------------------------------

if "period" not in df.columns:
    raise ValueError("The 'period' column was not found.")

period_summary = (
    df.groupby("period", as_index=False)[FEATURE_COLS]
    .mean()
    .sort_values("period")
)

period_summary.to_csv(
    OUTPUT_DIR / "chapter_period_feature_means.csv",
    index=False,
)

n_features = len(FEATURE_COLS)
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(12, 4 * n_rows),
    sharex=True,
)

axes = np.asarray(axes).reshape(-1)

for ax, col in zip(axes, FEATURE_COLS):
    ax.plot(
        period_summary["period"],
        period_summary[col],
        linewidth=2.5,
    )

    ax.set_title(
        FEATURE_LABELS.get(col, col),
        fontsize=SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    style_axis(
        ax,
        xlabel="Half-hour trading period",
        ylabel="Mean value",
    )

for ax in axes[len(FEATURE_COLS):]:
    ax.axis("off")

save_current_figure("fig_mean_features_by_period.png")


# ------------------------------------------------------------
# 9. rolling means for all explanatory variables
# ------------------------------------------------------------

rolling_feature_columns = []

for col in FEATURE_COLS:
    rolling_col = f"{col}_rolling_mean"

    df[rolling_col] = (
        df[col]
        .rolling(
            ROLLING_FEATURE_WINDOW,
            min_periods=max(1, ROLLING_FEATURE_WINDOW // 5),
        )
        .mean()
    )

    rolling_feature_columns.append(rolling_col)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(12, 4 * n_rows),
    sharex=True,
)

axes = np.asarray(axes).reshape(-1)

for ax, col in zip(axes, FEATURE_COLS):
    rolling_col = f"{col}_rolling_mean"

    ax.plot(
        df["stream_index"],
        df[rolling_col],
        linewidth=2.5,
    )

    ax.set_title(
        FEATURE_LABELS.get(col, col),
        fontsize=SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    style_axis(
        ax,
        xlabel="Stream index",
        ylabel="Rolling mean",
    )

for ax in axes[len(FEATURE_COLS):]:
    ax.axis("off")

save_current_figure("fig_rolling_mean_features.png")


# ------------------------------------------------------------
# 10. Minimal early / middle / late summary table
# ------------------------------------------------------------

df["stream_segment"] = pd.qcut(
    df["stream_index"],
    q=3,
    labels=["early", "middle", "late"],
)

segment_summary = (
    df.groupby("stream_segment")
    .agg(
        n_observations=("stream_index", "count"),
        up_proportion=("class_binary", "mean"),
        mean_nswprice=("nswprice", "mean"),
        mean_nswdemand=("nswdemand", "mean"),
        mean_vicprice=("vicprice", "mean"),
        mean_vicdemand=("vicdemand", "mean"),
        mean_transfer=("transfer", "mean"),
    )
    .reset_index()
)

segment_summary.to_csv(
    OUTPUT_DIR / "chapter_segment_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 11. Finish
# ------------------------------------------------------------

print("\nChapter-focused ELEC2 EDA complete.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")

print("\nSaved files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {file.name}")