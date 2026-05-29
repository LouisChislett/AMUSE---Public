import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from stable_baselines3 import PPO

from RealEnvironment import ModelUpdatingEnv

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)


# =============================================================================
# Configuration
# =============================================================================
SEED = 144
np.random.seed(SEED)

BASE_PATH = r"PATH"
OUTPUT_DIR = "Outputs"
MODEL_DIR = "Models"

START_YEAR = 2012
END_YEAR = 2022
LAST_MONTH_IN_END_YEAR = 10

COSTS = [0.05, 0.4]

RL_CONFIGS = {
    0.05: {
        "model_path": os.path.join(MODEL_DIR, "best_model_0.05.zip"),
        "env_update_penalty": 0.05,
    },
    0.4: {
        "model_path": os.path.join(MODEL_DIR, "best_model_0.4.zip"),
        "env_update_penalty": 0.05,
    },
}

BASE_CLASSIFIER = LogisticRegression(
    random_state=SEED,
    max_iter=2000,
    solver="lbfgs",
)


# =============================================================================
# Metric helpers
# =============================================================================
def safe_auc_score(y_true, y_score):
    """Compute ROC AUC, returning 0.5 when AUC is undefined."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5


def evaluate_classifier(classifier, X, y):
    """Return accuracy and AUC for a fitted classifier on one batch."""
    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = safe_auc_score(y, y_proba)

    return accuracy, auc, y_pred


def fresh_classifier():
    """Return a fresh estimator so experiments cannot leak fitted state."""
    return clone(BASE_CLASSIFIER)


# =============================================================================
# SPARRA data loading
# =============================================================================
def encode_binary_target(y, month_label):
    """Robustly coerce a binary target to 0/1."""
    y = pd.Series(y).copy()

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y_str = y.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
        }

        if set(y_str.dropna().unique()).issubset(set(mapping.keys())):
            y = y_str.map(mapping)
        else:
            codes, uniques = pd.factorize(y_str)
            if len(uniques) != 2:
                raise ValueError(f"{month_label}: target is not binary. Classes: {list(uniques)}")
            y = pd.Series(codes, index=y.index)

    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():
        raise ValueError(f"{month_label}: target contains values that cannot be converted to binary.")

    y = y.astype(int)

    if len(np.unique(y)) != 2:
        raise ValueError(f"{month_label}: target has {len(np.unique(y))} class(es); expected 2.")

    return y


def load_sparra_monthly_datasets(base_path=BASE_PATH):
    """
    Load SPARRA monthly CSV files and return:
      - datasets: list of (X_numpy, y_numpy)
      - month_labels: list of YYYY-MM strings

    Important:
    Features are first collected as DataFrames and then aligned to a common
    column set. This avoids the common one-hot encoding bug where one month has
    a different set/order of decile dummy columns than another month.
    """
    print("Loading monthly SPARRA datasets...")

    feature_frames = []
    target_arrays = []
    month_labels = []

    for year in range(START_YEAR, END_YEAR + 1):
        max_month = LAST_MONTH_IN_END_YEAR if year == END_YEAR else 12

        for month in range(1, max_month + 1):
            month_label = f"{year}-{month:02d}"
            csv_path = os.path.join(base_path, f"AMUSE_{month_label}.csv")

            try:
                df = pd.read_csv(csv_path)

                required_cols = ["target", "age", "decile", "sexM"]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    print(f"  ✗ {month_label}: missing columns {missing}")
                    continue

                df = df[required_cols].copy()
                df["age"] = pd.to_numeric(df["age"], errors="coerce")
                df["sexM"] = pd.to_numeric(df["sexM"], errors="coerce")
                df["decile"] = df["decile"].astype(str).str.strip()

                df = df.dropna(subset=required_cols).copy()
                if len(df) == 0:
                    print(f"  ✗ {month_label}: no valid rows after cleaning")
                    continue

                y = encode_binary_target(df["target"], month_label)

                X = pd.DataFrame({
                    "age": df["age"].astype(float),
                    "sexM": df["sexM"].astype(int),
                }, index=df.index)

                decile_dummies = pd.get_dummies(df["decile"], prefix="decile", drop_first=True)
                X = pd.concat([X, decile_dummies], axis=1)

                feature_frames.append(X)
                target_arrays.append(y.to_numpy())
                month_labels.append(month_label)

                print(f"  ✓ {month_label}: loaded {len(df)} samples")

            except FileNotFoundError:
                print(f"  ✗ {month_label}: file not found")
            except Exception as exc:
                print(f"  ✗ {month_label}: error - {exc}")

    if not feature_frames:
        raise RuntimeError("No monthly SPARRA datasets loaded successfully.")

    # Align all monthly feature matrices to the union of columns in a stable order.
    all_columns = sorted(set().union(*(frame.columns for frame in feature_frames)))
    aligned_features = [
        frame.reindex(columns=all_columns, fill_value=0).astype(float).to_numpy()
        for frame in feature_frames
    ]

    datasets = list(zip(aligned_features, target_arrays))

    print(f"\nSuccessfully loaded {len(datasets)} monthly datasets")
    print(f"Date range: {month_labels[0]} to {month_labels[-1]}")
    print(f"Number of aligned features: {len(all_columns)}\n")

    return datasets, month_labels


# =============================================================================
# Baseline experiments
# =============================================================================
def run_always_update_experiment(datasets):
    """Fit on every batch before evaluating that batch, matching the original script."""
    classifier = fresh_classifier()

    accuracies = []
    aucs = []
    update_steps = []

    for i, (X_batch, y_batch) in enumerate(datasets):
        classifier.fit(X_batch, y_batch)
        if i > 0:
            update_steps.append(i)

        accuracy, auc, _ = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

    return accuracies, aucs, update_steps


def run_never_update_experiment(datasets):
    """Fit on the first batch only, then evaluate all batches."""
    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    accuracies = []
    aucs = []

    for X_batch, y_batch in datasets:
        accuracy, auc, _ = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

    return accuracies, aucs, []


def run_random_update_experiment(datasets, update_prob, seed=SEED):
    """Randomly refit before evaluating the current batch, as in the original script."""
    rng = np.random.default_rng(seed)
    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    accuracies = []
    aucs = []
    update_steps = []

    # Initial batch.
    accuracy, auc, _ = evaluate_classifier(classifier, datasets[0][0], datasets[0][1])
    accuracies.append(accuracy)
    aucs.append(auc)

    for i in range(1, len(datasets)):
        X_batch, y_batch = datasets[i]

        if rng.uniform() < update_prob:
            classifier.fit(X_batch, y_batch)
            update_steps.append(i)

        accuracy, auc, _ = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

    return accuracies, aucs, update_steps


def compute_even_update_steps(num_datasets, target_num_updates):
    """
    Choose exactly target_num_updates indices from 1..num_datasets-1, spread evenly.
    """
    target_num_updates = int(target_num_updates)

    if target_num_updates <= 0:
        return []

    decision_points = np.arange(1, num_datasets)

    if target_num_updates >= len(decision_points):
        return decision_points.tolist()

    positions = np.linspace(0, len(decision_points) - 1, target_num_updates)
    update_steps = sorted(set(decision_points[np.rint(positions).astype(int)].tolist()))

    # Fill any duplicates caused by rounding.
    if len(update_steps) < target_num_updates:
        remaining = [step for step in decision_points.tolist() if step not in update_steps]
        update_steps = sorted(update_steps + remaining[: target_num_updates - len(update_steps)])

    return update_steps


def run_even_update_experiment(datasets, target_num_updates):
    """Refit at exactly target_num_updates evenly spaced batches."""
    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    update_steps = compute_even_update_steps(len(datasets), target_num_updates)
    update_step_set = set(update_steps)

    accuracies = []
    aucs = []

    accuracy, auc, _ = evaluate_classifier(classifier, datasets[0][0], datasets[0][1])
    accuracies.append(accuracy)
    aucs.append(auc)

    for i in range(1, len(datasets)):
        X_batch, y_batch = datasets[i]

        if i in update_step_set:
            classifier.fit(X_batch, y_batch)

        accuracy, auc, _ = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

    return accuracies, aucs, update_steps


def run_ddm_experiment(datasets, warning_level=2.0, drift_level=3.0):
    """
    Batch adaptation of DDM (Drift Detection Method).

    Evaluation order:
      1. evaluate the current batch with the current model;
      2. compute the current DDM statistic;
      3. test for drift against the previously stored historical minimum;
      4. update the historical minimum only after the drift test;
      5. if drift is detected, refit on the current batch for future batches.
    """
    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    accuracies = []
    aucs = []
    update_steps = []

    total_errors = 0
    total_instances = 0
    p_min = np.inf
    s_min = np.inf

    for i, (X_batch, y_batch) in enumerate(datasets):
        accuracy, auc, predictions = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

        batch_errors = int(np.sum(predictions != y_batch))
        total_errors += batch_errors
        total_instances += len(y_batch)

        p = total_errors / total_instances
        s = np.sqrt(p * (1 - p) / total_instances)

        drift_detected = (i > 0) and ((p + s) >= (p_min + drift_level * s_min))

        # Update the historical minimum only after the current drift test.
        if p + s <= p_min + s_min:
            p_min = p
            s_min = s

        if drift_detected:
            classifier.fit(X_batch, y_batch)
            update_steps.append(i)

            # Reset detector after model update. The next batch starts a new
            # post-update monitoring period.
            total_errors = 0
            total_instances = 0
            p_min = np.inf
            s_min = np.inf

    return accuracies, aucs, update_steps


def run_frd_experiment(
    datasets,
    alpha=0.004,
    epsilon=1e-12,
):
    """
    Principled batched adaptation of Four Rates Detector (FRD).

    FRD monitors four classifier-performance rates:
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)

    For each rate, the detector compares the current batch rate with the
    denominator-weighted post-reset reference rate using a one-sided
    two-proportion z-test. A positive z-statistic indicates deterioration:
    the current rate is lower than the reference rate.

    Drift is declared if any defined and testable rate is significantly lower
    than its reference value, using a Bonferroni correction across the rates
    that are testable at the current batch.

    Evaluation order:
      1. fit the initial classifier on batch 0;
      2. predict/evaluate the current batch using the current classifier;
      3. compute the current batch confusion-matrix rates;
      4. compare current rates with post-reset reference rates from previous
         accepted batches only;
      5. if drift is detected, retrain on the current batch and reset the
         detector state;
      6. otherwise, add the current batch to the post-reset reference state.
    """
    from scipy.stats import norm

    if len(datasets) == 0:
        return [], [], []

    rate_names = ["TPR", "TNR", "PPV", "NPV"]

    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    accuracies = []
    aucs = []
    update_steps = []

    # Contains confusion-matrix counts for accepted batches since the most
    # recent detector reset. The current batch is not added until after testing.
    reference_counts = []

    def confusion_counts(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def rate_success_denominator(counts, rate_name):
        tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

        if rate_name == "TPR":
            return tp, tp + fn
        if rate_name == "TNR":
            return tn, tn + fp
        if rate_name == "PPV":
            return tp, tp + fp
        if rate_name == "NPV":
            return tn, tn + fn

        raise ValueError(f"Unknown FRD rate: {rate_name}")

    def aggregate_counts(counts_list):
        return {
            key: int(sum(counts[key] for counts in counts_list))
            for key in ["tp", "tn", "fp", "fn"]
        }

    for i, (X_batch, y_batch) in enumerate(datasets):
        accuracy, auc, predictions = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

        current_counts = confusion_counts(y_batch, predictions)
        drift_detected = False
        z_statistics = []

        # Test only after at least one accepted post-reset reference batch exists.
        # The current batch is deliberately excluded from the reference.
        if i > 0 and len(reference_counts) > 0:
            reference_total = aggregate_counts(reference_counts)

            for rate_name in rate_names:
                current_successes, current_denominator = rate_success_denominator(
                    current_counts, rate_name
                )
                reference_successes, reference_denominator = rate_success_denominator(
                    reference_total, rate_name
                )

                # Skip rates that have no samples or undefined denominators.
                if current_denominator <= 0 or reference_denominator <= 0:
                    continue

                q_current = current_successes / current_denominator
                q_reference = reference_successes / reference_denominator

                # Pooled estimate under the null that the current and reference
                # rates are equal.
                q_pooled = (
                    current_successes + reference_successes
                ) / (
                    current_denominator + reference_denominator
                )

                standard_error = np.sqrt(
                    q_pooled
                    * (1.0 - q_pooled)
                    * (1.0 / current_denominator + 1.0 / reference_denominator)
                )

                if standard_error <= epsilon:
                    continue

                # Positive z means deterioration: the current rate is lower
                # than the denominator-weighted reference rate.
                z_statistic = (q_reference - q_current) / standard_error
                z_statistics.append(z_statistic)

            if z_statistics:
                # Bonferroni correction over the rates that are actually
                # testable in the current batch.
                threshold = norm.ppf(1.0 - alpha / len(z_statistics))
                drift_detected = any(z > threshold for z in z_statistics)

        if drift_detected:
            classifier.fit(X_batch, y_batch)
            update_steps.append(i)

            # Reset detector state after adaptation. The drift-triggering batch
            # is used to update the classifier, but not to initialise the new
            # reference period; monitoring restarts from the next batch.
            reference_counts = []
        else:
            # Accept the current batch as belonging to the current post-reset
            # concept and include it in future reference estimates.
            reference_counts.append(current_counts)

    return accuracies, aucs, update_steps


def run_stepd_experiment(
    datasets,
    alpha=0.003,
):
    """
    Batch Statistical Test of Equal Proportions Drift Detection (STEPD).

    STEPD compares the classifier's current batch accuracy with its reference
    accuracy using a one-sided two-sample test for a difference in proportions.
    The detector stores instance-level binary correctness values for accepted
    post-reset batches. At each step, the current batch forms the recent sample,
    and all previous accepted post-reset batches form the reference sample.

    Drift is declared when current accuracy is significantly lower than reference
    accuracy. After drift, the classifier is retrained on the current batch and
    the detector state is reset for future batches.
    """
    from scipy.stats import norm

    classifier = fresh_classifier()
    classifier.fit(datasets[0][0], datasets[0][1])

    accuracies = []
    aucs = []
    update_steps = []

    correctness_batches = []

    for i, (X_batch, y_batch) in enumerate(datasets):
        accuracy, auc, predictions = evaluate_classifier(classifier, X_batch, y_batch)
        accuracies.append(accuracy)
        aucs.append(auc)

        batch_correctness = (predictions == y_batch).astype(int)
        drift_detected = False

        # Test only after at least one post-reset reference batch exists.
        if len(correctness_batches) > 0 and i > 0:
            reference_correctness = np.concatenate(correctness_batches)
            recent_correctness = batch_correctness

            n_ref = len(reference_correctness)
            n_recent = len(recent_correctness)

            if n_ref > 0 and n_recent > 0:
                a_ref = np.mean(reference_correctness)
                a_recent = np.mean(recent_correctness)
                a_pooled = (
                    np.sum(reference_correctness) + np.sum(recent_correctness)
                ) / (n_ref + n_recent)

                standard_error = np.sqrt(
                    a_pooled
                    * (1.0 - a_pooled)
                    * (1.0 / n_ref + 1.0 / n_recent)
                )

                if standard_error > 0:
                    z_statistic = (a_ref - a_recent) / standard_error
                    threshold = norm.ppf(1.0 - alpha)
                    drift_detected = z_statistic > threshold

        if drift_detected:
            classifier.fit(X_batch, y_batch)
            update_steps.append(i)
            correctness_batches = []
        else:
            # Add current batch to reference pool only if drift was not detected.
            correctness_batches.append(batch_correctness)

    return accuracies, aucs, update_steps


# =============================================================================
# RL experiment
# =============================================================================
def run_rl_experiment(datasets, model_path, env_update_penalty):
    """Run a trained PPO model on the SPARRA monthly sequence."""
    n_timesteps = len(datasets)

    model = PPO.load(model_path)

    np.random.seed(SEED)
    env = ModelUpdatingEnv(
        datasets,
        update_penalty=env_update_penalty,
        n_timesteps=n_timesteps,
    )

    obs = env.reset()
    done = False

    for _ in range(len(datasets) - 1):
        if done:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    accuracy_log = env.get_accuracy_log()
    update_steps = env.get_update_steps()

    auc_log = env.get_auc_log()

    # Remove the initial training entry if the environment logs it as an update.
    if len(update_steps) > 0 and update_steps[0] == 0:
        update_steps = update_steps[1:]

    # Defensive checks: the environment should have one metric value per month.
    if len(accuracy_log) != len(datasets):
        raise RuntimeError(
            f"RL accuracy log length is {len(accuracy_log)}, but expected {len(datasets)}. "
            "Check RealEnvironment.reset()/step() logging."
        )
    if len(auc_log) != len(datasets):
        raise RuntimeError(
            f"RL AUC log length is {len(auc_log)}, but expected {len(datasets)}. "
            "Check RealEnvironment.get_auc_log() logging."
        )

    return accuracy_log, auc_log, update_steps


# =============================================================================
# Results assembly / saving
# =============================================================================
def build_utility_table(results_by_cost, metric_key="auc_scores"):
    """Utility is cumulative metric minus penalty times number of updates."""
    utilities_dict = {}

    for penalty in COSTS:
        metric_results = results_by_cost[metric_key][penalty]
        update_steps = results_by_cost["update_steps"][penalty]

        utilities_dict[penalty] = {
            method: np.sum(metric_results[method]) - penalty * len(update_steps[method])
            for method in metric_results
        }

    utilities = pd.DataFrame(utilities_dict).T
    utilities.index.name = "Update Penalty"
    return utilities, utilities_dict


def save_average_metric_csv(results_by_cost, metric_key, metric_label):
    """Save one average metric CSV per cost."""
    for penalty in COSTS:
        averages = {
            method: np.mean(values)
            for method, values in results_by_cost[metric_key][penalty].items()
        }

        df = pd.DataFrame.from_dict(averages, orient="index", columns=[metric_label])
        suffix = "005" if penalty == 0.05 else "04"
        output_path = os.path.join(OUTPUT_DIR, f"average_{metric_key[:-1]}_cost_{suffix}.csv")
        df.to_csv(output_path)
        print(f"✓ Saved: {output_path}")


def main():
    datasets, month_labels = load_sparra_monthly_datasets()
    n_timesteps = len(datasets)

    print("Running RL agents on monthly SPARRA sequence...")

    rl_results = {}

    for penalty in COSTS:
        config = RL_CONFIGS[penalty]
        acc_log, auc_log, update_steps = run_rl_experiment(
            datasets,
            model_path=config["model_path"],
            env_update_penalty=config["env_update_penalty"],
        )

        rl_results[penalty] = {
            "accuracies": acc_log,
            "auc_scores": auc_log,
            "update_steps": update_steps,
            "prob": len(update_steps) / n_timesteps,
            "num_updates": len(update_steps),
        }

        print(
            f"  Cost {penalty}: {len(update_steps)} updates "
            f"(prob={len(update_steps) / n_timesteps:.4f})"
        )

    print("\nRunning penalty-invariant baselines...")

    acc_always, auc_always, steps_always = run_always_update_experiment(datasets)
    acc_never, auc_never, steps_never = run_never_update_experiment(datasets)
    acc_ddm, auc_ddm, steps_ddm = run_ddm_experiment(datasets)
    acc_frd, auc_frd, steps_frd = run_frd_experiment(datasets)
    acc_stepd, auc_stepd, steps_stepd = run_stepd_experiment(datasets)

    print("✓ Penalty-invariant baselines completed")

    print("\nRunning penalty-specific random/even baselines...")

    random_even_results = {}

    for penalty in COSTS:
        prob = rl_results[penalty]["prob"]
        num_updates = rl_results[penalty]["num_updates"]

        acc_random, auc_random, steps_random = run_random_update_experiment(
            datasets,
            update_prob=prob,
            seed=SEED,
        )

        acc_even, auc_even, steps_even = run_even_update_experiment(
            datasets,
            target_num_updates=num_updates,
        )

        random_even_results[penalty] = {
            "Random": {
                "accuracies": acc_random,
                "auc_scores": auc_random,
                "update_steps": steps_random,
            },
            "Even": {
                "accuracies": acc_even,
                "auc_scores": auc_even,
                "update_steps": steps_even,
            },
        }

        print(
            f"  Cost {penalty}: "
            f"AMUSE={num_updates}, "
            f"Random={len(steps_random)} (p={prob:.4f}), "
            f"Even={len(steps_even)}"
        )

    all_results = {
        "accuracies": {},
        "auc_scores": {},
        "update_steps": {},
        "num_updates": {},
    }

    for penalty in COSTS:
        all_results["accuracies"][penalty] = {
            "AMUSE": rl_results[penalty]["accuracies"],
            "Always Update": acc_always,
            "Never Update": acc_never,
            "Random": random_even_results[penalty]["Random"]["accuracies"],
            "Even": random_even_results[penalty]["Even"]["accuracies"],
            "DDM": acc_ddm,
            "FRD": acc_frd,
            "STEPD": acc_stepd,
        }

        all_results["auc_scores"][penalty] = {
            "AMUSE": rl_results[penalty]["auc_scores"],
            "Always Update": auc_always,
            "Never Update": auc_never,
            "Random": random_even_results[penalty]["Random"]["auc_scores"],
            "Even": random_even_results[penalty]["Even"]["auc_scores"],
            "DDM": auc_ddm,
            "FRD": auc_frd,
            "STEPD": auc_stepd,
        }

        all_results["update_steps"][penalty] = {
            "AMUSE": rl_results[penalty]["update_steps"],
            # Use the actual refit decisions after initial training.
            "Always Update": steps_always,
            "Never Update": steps_never,
            "Random": random_even_results[penalty]["Random"]["update_steps"],
            "Even": random_even_results[penalty]["Even"]["update_steps"],
            "DDM": steps_ddm,
            "FRD": steps_frd,
            "STEPD": steps_stepd,
        }

        all_results["num_updates"][penalty] = {
            method: len(steps)
            for method, steps in all_results["update_steps"][penalty].items()
        }

    print("\nCalculating utilities based on AUC...")
    utilities, utilities_dict = build_utility_table(all_results, metric_key="auc_scores")
    all_results["average_total_utility"] = utilities_dict

    # Compatibility aliases used by existing plotting scripts.
    all_results["average_updates"] = all_results["num_updates"][0.05]
    all_results["months"] = month_labels

    print("\nCumulative utilities across all months, based on AUC:")
    print(utilities)
    print()

    for penalty in COSTS:
        print(f"Number of updates for each method, cost penalty = {penalty}:")
        for method, updates in all_results["num_updates"][penalty].items():
            print(f"  {method}: {updates}")
        print()

    print(f"Saving results to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "results.npy"), all_results)
    print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'results.npy')}")

    utilities.T.to_csv(os.path.join(OUTPUT_DIR, "utilities.csv"))
    print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'utilities.csv')}")

    for penalty in COSTS:
        suffix = "005" if penalty == 0.05 else "04"

        num_updates_df = pd.DataFrame.from_dict(
            all_results["num_updates"][penalty],
            orient="index",
            columns=["Number of Updates"],
        )
        num_updates_path = os.path.join(OUTPUT_DIR, f"num_updates_cost_{suffix}.csv")
        num_updates_df.to_csv(num_updates_path)
        print(f"✓ Saved: {num_updates_path}")

    print("\nCalculating and saving average metrics...")
    save_average_metric_csv(all_results, "auc_scores", "Average AUC")
    save_average_metric_csv(all_results, "accuracies", "Average Accuracy")

    print("\nDone.")


if __name__ == "__main__":
    main()
