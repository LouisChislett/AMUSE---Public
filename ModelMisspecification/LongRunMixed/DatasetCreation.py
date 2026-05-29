from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

SEED = 144
N_SAMPLES = 1000
N_OBSERVED_FEATURES = 5
N_TIMESTEPS = 50_000
N_DATASET_REPLICATES = 1
OUTPUT_DIR = "Datasets"

# The true data-generating process clips coefficients over a wider range than
# the AMUSE simulator, which clips its five linear coefficients to [-1, 1].
TRUE_COEF_LOW = -2.0
TRUE_COEF_HIGH = 2.0

# Prevent overflow in the sigmoid for large nonlinear logits.
LOGIT_CLIP = 20.0


@dataclass(frozen=True)
class DriftRegime:
    """Piecewise nonstationary true drift regime."""

    end_time: int
    drift_strength: float
    sudden_drift_prob: float
    sign_flip_prob: float
    drift_label: str

DRIFT_REGIMES = (
    DriftRegime(end_time=10_000, drift_strength=0.04, sudden_drift_prob=0.00, sign_flip_prob=0.00, drift_label="mild_gradual"),
    DriftRegime(end_time=20_000, drift_strength=0.08, sudden_drift_prob=0.03, sign_flip_prob=0.01, drift_label="moderate_abrupt"),
    DriftRegime(end_time=35_000, drift_strength=0.15, sudden_drift_prob=0.08, sign_flip_prob=0.03, drift_label="severe_mixed"),
    DriftRegime(end_time=N_TIMESTEPS + 1, drift_strength=0.06, sudden_drift_prob=0.15, sign_flip_prob=0.05, drift_label="frequent_regime_switch"),
)


# -----------------------------------------------------------------------------
# True data-generating model
# -----------------------------------------------------------------------------

COEFFICIENT_NAMES = (
    "intercept",
    "x1", "x2", "x3", "x4", "x5",
    "x1_squared", "x2_squared",
    "x2_x3_interaction", "x4_x5_interaction", "x1_x5_interaction",
    "sin_x3", "sin_x4",
    "indicator_x1_positive", "indicator_x5_large",
    "hidden_z1", "hidden_z2",
)

N_TRUE_FEATURES = len(COEFFICIENT_NAMES)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    logits = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-logits))


def make_true_design_matrix(
    X_observed: np.ndarray,
    rng: np.random.Generator,
    hidden_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the richer true design matrix used internally to generate Y.

    Only X_observed is returned in the final datasets. The hidden features and
    nonlinear transformations are unavailable to the fitted classifier and to
    AMUSE, creating model misspecification.
    """
    if hidden_features is None:
        hidden_features = rng.normal(loc=0.0, scale=1.0, size=(X_observed.shape[0], 2))

    x1, x2, x3, x4, x5 = (X_observed[:, j] for j in range(5))
    z1, z2 = hidden_features[:, 0], hidden_features[:, 1]

    design = np.column_stack(
        [
            np.ones(X_observed.shape[0]),
            x1,
            x2,
            x3,
            x4,
            x5,
            x1**2,
            x2**2,
            x2 * x3,
            x4 * x5,
            x1 * x5,
            np.sin(x3),
            np.sin(x4),
            (x1 > 0.0).astype(float),
            (x5 > 1.0).astype(float),
            z1,
            z2,
        ]
    )
    return design, hidden_features


def draw_initial_true_coefficients(rng: np.random.Generator) -> np.ndarray:
    """
    Draw initial coefficients with enough observable signal that model updating
    can still help, while preserving substantial hidden/nonlinear signal.
    """
    beta = np.zeros(N_TRUE_FEATURES)

    # Intercept keeps the class balance from becoming too extreme initially.
    beta[0] = rng.uniform(-0.75, 0.75)

    # Observable linear terms: the classifier can partially recover these.
    beta[1:6] = rng.uniform(-0.75, 0.75, size=5)

    # Nonlinear and interaction terms: unavailable to the downstream classifier.
    beta[6:15] = rng.uniform(-0.75, 0.75, size=9)

    # Hidden confounding terms: used to generate Y but not observed by AMUSE or C.
    beta[15:17] = rng.uniform(-0.75, 0.75, size=2)

    return beta


def get_regime(t: int) -> DriftRegime:
    """Return the active true drift regime at time t."""
    for regime in DRIFT_REGIMES:
        if t < regime.end_time:
            return regime
    return DRIFT_REGIMES[-1]


def drift_true_coefficients(
    previous_beta: np.ndarray,
    t: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Nonstationary true coefficient drift.

    This deliberately differs from ModelUpdatingEnv.py, which uses stationary
    five-dimensional Gaussian drift with scale 0.03 and no sudden changes.
    """
    regime = get_regime(t)

    if rng.random() < regime.sudden_drift_prob:
        # Abrupt regime jump. Preserve a reasonable amount of linear signal but
        # allow nonlinear/hidden effects to be larger and more volatile.
        beta = np.empty_like(previous_beta)
        beta[0] = rng.uniform(-0.50, 0.50)
        beta[1:6] = rng.uniform(-1.25, 1.25, size=5)
        beta[6:15] = rng.uniform(TRUE_COEF_LOW, TRUE_COEF_HIGH, size=9)
        beta[15:17] = rng.uniform(-1.50, 1.50, size=2)
    else:
        drift = rng.normal(loc=0.0, scale=regime.drift_strength, size=previous_beta.shape)

        # Make nonlinear and hidden effects drift more than observable linear
        # effects. This increases simulator misspecification without making
        # retraining entirely useless.
        drift[6:15] *= 1
        drift[15:17] *= 1.5

        beta = previous_beta + drift

    # Occasional sign flips create abrupt changes that are not equivalent to a
    # small random walk. They are particularly challenging for a static policy.
    if rng.random() < regime.sign_flip_prob:
        affected = rng.choice(np.arange(1, N_TRUE_FEATURES), size=rng.integers(2, 6), replace=False)
        beta[affected] *= -1.0

    return np.clip(beta, TRUE_COEF_LOW, TRUE_COEF_HIGH)


def generate_labels(
    X_observed: np.ndarray,
    beta: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and binary labels from the true DGP."""
    X_true, _ = make_true_design_matrix(X_observed, rng)
    probs = sigmoid(X_true @ beta)
    Y = rng.binomial(1, probs)
    return probs, Y


def generate_observed_features(
    rng: np.random.Generator,
    t: int | None = None,
) -> np.ndarray:
    """
    Generate observed covariates.

    The marginal X distribution is kept approximately stationary to remain close
    to the paper's focus on real concept drift, i.e. changes in Y|X rather than
    major covariate shift.
    """
    return rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, N_OBSERVED_FEATURES))


def generate_drifted_datasets(
    n_timesteps: int,
    n_features: int,
    n_combined_features: int,
    X: np.ndarray,
    X_combined: np.ndarray,
    initial_real_coefficients: np.ndarray,
    probs: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate one full drifted evaluation episode.

    Parameters are kept for backwards compatibility with the original script.
    The true coefficient vector is intentionally longer than n_features because
    the real data-generating process uses nonlinear and hidden terms.
    """
    del n_features, n_combined_features, X_combined  # kept only for compatibility

    rng = np.random.default_rng(seed)

    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    beta = initial_real_coefficients.copy()

    # Store initial dataset. Re-draw Y from the supplied initial probabilities so
    # each replicate can differ even when sharing the same initial X.
    Y0 = rng.binomial(1, probs)
    dataset.append((X.copy(), Y0))

    for t in range(1, n_timesteps):
        beta = drift_true_coefficients(beta, t, rng)
        X_t = generate_observed_features(rng, t)
        _, Y_t = generate_labels(X_t, beta, rng)
        dataset.append((X_t, Y_t))

    return dataset


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

rng_main = np.random.default_rng(SEED)

# Generate initial observed data.
X = generate_observed_features(rng_main, t=0)
initial_real_coefficients = draw_initial_true_coefficients(rng_main)
X_combined, _ = make_true_design_matrix(X, rng_main)
n_combined_features = X_combined.shape[1]

# Initial probabilities and initial labels.
probs = sigmoid(X_combined @ initial_real_coefficients)
Y = rng_main.binomial(1, probs)
initial_dataset = (X, Y)

# Generate drifted datasets with different seeds.
seeds = rng_main.integers(0, 1_000_000, size=N_DATASET_REPLICATES)
datasets = {}

for i, seed in enumerate(seeds):
    datasets[i] = generate_drifted_datasets(
        N_TIMESTEPS,
        N_OBSERVED_FEATURES,
        n_combined_features,
        X,
        X_combined,
        initial_real_coefficients,
        probs,
        int(seed),
    )

# Save outputs using the same filenames as the original pipeline.
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.savez(os.path.join(OUTPUT_DIR, "initial_dataset.npz"), X=X, Y=Y)
np.save(os.path.join(OUTPUT_DIR, "drifted_datasets.npy"), datasets, allow_pickle=True)
np.save(os.path.join(OUTPUT_DIR, "initial_real_coefficients.npy"), initial_real_coefficients)
np.save(os.path.join(OUTPUT_DIR, "probs.npy"), probs)

# Extra metadata is useful for the paper/supplement and does not break the old
# pipeline because it is saved as an additional file.
np.savez(
    os.path.join(OUTPUT_DIR, "severe_misspecification_metadata.npz"),
    coefficient_names=np.array(COEFFICIENT_NAMES),
    drift_regimes=np.array(
        [
            (r.end_time, r.drift_strength, r.sudden_drift_prob, r.sign_flip_prob, r.drift_label)
            for r in DRIFT_REGIMES
        ],
        dtype=object,
    ),
    true_coef_low=TRUE_COEF_LOW,
    true_coef_high=TRUE_COEF_HIGH,
    seed=SEED,
)
