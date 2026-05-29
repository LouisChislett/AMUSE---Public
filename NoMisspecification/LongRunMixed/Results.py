import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO
import os
from scipy.stats import norm
from sklearn.base import clone

from RealEnvironment import ModelUpdatingEnv

# set the seed
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

# Load the datasets
all_datasets = np.load("Datasets/drifted_datasets.npy", allow_pickle=True).item()

n_timesteps = 50000

# Define the update penalties
costs = [0.05, 0.4]

# # Reinforcement Learning with Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_transfer_005 = {}
update_steps_results_RL_transfer_005 = {}

# Run the model for each set of drifted datasets
for key in all_datasets:
    np.random.seed(144)
    # Load the pre-trained PPO model
    pretrained_model = PPO.load(f'Models/best_model_0.05.zip')

    # Set up the environment for each set of drifted datasets
    drifted_datasets = all_datasets[key]
    np.random.seed(144)
    env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.02, n_timesteps=n_timesteps)
    
    # Reinitialize a new PPO model with the same policy but custom parameters
    model = PPO('MlpPolicy', env, 
                n_steps=1000,            # Number of steps to collect before updating the model
                batch_size=50,           # Set new batch size
                verbose=1,
                gamma=0.8)
    
    # Copy the policy from the pre-trained model to the new model
    model.policy = pretrained_model.policy

    # Train the model for the one episode
    model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)

    # Collect the accuracy log and update steps for this set
    accuracy_log_005 = env.get_accuracy_log()
    update_steps_005 = env.get_update_steps()

    # Store the results in the dictionary with the key
    accuracies_results_RL_transfer_005[key] = accuracy_log_005
    update_steps_results_RL_transfer_005[key] = update_steps_005


#remove first update step from the update steps results (as we don't include the initial model)
for key in update_steps_results_RL_transfer_005:
    update_steps_results_RL_transfer_005[key] = update_steps_results_RL_transfer_005[key][1:]

# find the probability of updating the model (used for random updates later)
average_updates_RL_transfer_005 = np.mean([len(update_steps_results_RL_transfer_005[key]) for key in update_steps_results_RL_transfer_005])
prob_005 = average_updates_RL_transfer_005/ n_timesteps


# # Reinforcement Learning without Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_005 = {}
update_steps_results_RL_005 = {}

# Run the model for each set of drifted datasets
for key in all_datasets:
    # Load the pre-trained model
    model = PPO.load(f'Models/best_model_0.05.zip')
    drifted_datasets = all_datasets[key]

    # Set up the environment for each set of drifted datasets
    np.random.seed(144)
    env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.02, n_timesteps=n_timesteps)
    
    obs = env.reset()
    done = False

    # Run the environment until done
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    # Collect the accuracy log and update steps for this set
    accuracy_log_005 = env.get_accuracy_log()
    update_steps_005 = env.get_update_steps()

    # Store the results in the dictionary with the key
    accuracies_results_RL_005[key] = accuracy_log_005
    update_steps_results_RL_005[key] = update_steps_005


# remove first update step from the update steps results (as we don't include the initial model)
for key in update_steps_results_RL_005:
    update_steps_results_RL_005[key] = update_steps_results_RL_005[key][1:]


# # Reinforcement Learning with Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_transfer_04 = {}
update_steps_results_RL_transfer_04 = {}

# Run the model for each set of drifted datasets
for key in all_datasets:
    np.random.seed(144)
    # Load the pre-trained PPO model
    pretrained_model = PPO.load(f'Models/best_model_0.4.zip')

    # Set up the environment for each set of drifted datasets
    drifted_datasets = all_datasets[key]
    np.random.seed(144)
    env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.04, n_timesteps=n_timesteps)
    
    # Reinitialize a new PPO model with the same policy but custom parameters
    model = PPO('MlpPolicy', env, 
                n_steps=1000,            # Number of steps to collect before updating the model
                batch_size=50,           # Set new batch size
                verbose=1,
                gamma=0.8)
    
    # Copy the policy from the pre-trained model to the new model
    model.policy = pretrained_model.policy

    # Train the model for the one episode
    model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)

    # Collect the accuracy log and update steps for this set
    accuracy_log_04 = env.get_accuracy_log()
    update_steps_04 = env.get_update_steps()

    # Store the results in the dictionary with the key
    accuracies_results_RL_transfer_04[key] = accuracy_log_04
    update_steps_results_RL_transfer_04[key] = update_steps_04


#remove first update step from the update steps results (as we don't include the initial model)
for key in update_steps_results_RL_transfer_04:
    update_steps_results_RL_transfer_04[key] = update_steps_results_RL_transfer_04[key][1:]

# find the probability of updating the model (used for random updates later)
average_updates_RL_transfer_04 = np.mean([len(update_steps_results_RL_transfer_04[key]) for key in update_steps_results_RL_transfer_04])
prob_04 = average_updates_RL_transfer_04/ n_timesteps


# # Reinforcement Learning without Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_04 = {}
update_steps_results_RL_04 = {}

# Run the model for each set of drifted datasets
for key in all_datasets:
    # Load the pre-trained model
    model = PPO.load(f'Models/best_model_0.4.zip')
    drifted_datasets = all_datasets[key]

    # Set up the environment for each set of drifted datasets
    np.random.seed(144)
    env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.04, n_timesteps=n_timesteps)
    
    obs = env.reset()
    done = False

    # Run the environment until done
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    # Collect the accuracy log and update steps for this set
    accuracy_log_04 = env.get_accuracy_log()
    update_steps_04 = env.get_update_steps()

    # Store the results in the dictionary with the key
    accuracies_results_RL_04[key] = accuracy_log_04
    update_steps_results_RL_04[key] = update_steps_04


# remove first update step from the update steps results (as we don't include the initial model)
for key in update_steps_results_RL_04:
    update_steps_results_RL_04[key] = update_steps_results_RL_04[key][1:]


# define the classifier for other experiments
classifier = LogisticRegression()


# # Always Update

# Define the function to run the "always update" process on each dataset
def run_always_update_experiment(dataset, classifier):
    # Fit the model to the initial dataset
    classifier.fit(dataset[0][0], dataset[0][1])

    # Initialize a way of storing the accuracies
    accuracies_always = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0][0])
    accuracies_always.append(accuracy_score(dataset[0][1], initial_predictions))

    # Cycle through the datasets, applying updates and recording accuracies
    for i in range(1, len(dataset)):
        classifier.fit(dataset[i][0], dataset[i][1])
        predictions = classifier.predict(dataset[i][0])
        accuracies_always.append(accuracy_score(dataset[i][1], predictions))

    return accuracies_always

# Initialize storage for results
accuracies_results_always = {}

# Run the "always update" experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_always = run_always_update_experiment(dataset, classifier)
    accuracies_results_always[key] = accuracies_always

# # Never Update

# Define the function to run the "never update" process on each dataset
def run_never_update_experiment(dataset, classifier):

    # Fit the model to the initial dataset
    classifier.fit(dataset[0][0], dataset[0][1])

    # Initialize a way of storing the accuracies
    accuracies_never = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0][0])
    accuracies_never.append(accuracy_score(dataset[0][1], initial_predictions))

    # Cycle through the datasets without applying updates, recording accuracies
    for i in range(1, len(dataset)):
        predictions = classifier.predict(dataset[i][0])
        accuracies_never.append(accuracy_score(dataset[i][1], predictions))

    return accuracies_never

# Initialize storage for results
accuracies_results_never = {}

# Run the "never update" experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_never = run_never_update_experiment(dataset, classifier)
    accuracies_results_never[key] = accuracies_never

# # Random Updates
np.random.seed(144)
# Define the function to run the "random updates" process on each dataset
def run_random_update_experiment(dataset, classifier, update_prob=prob_005):

    # Fit the model to the initial dataset
    classifier.fit(dataset[0][0], dataset[0][1])

    # Initialize a way of storing the accuracies
    accuracies_random = []
    cels_random = []
    update_steps_random = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0][0])
    accuracies_random.append(accuracy_score(dataset[0][1], initial_predictions))

    # Cycle through the datasets, applying random updates and recording accuracies
    for i in range(1, len(dataset)):
        if np.random.uniform() < update_prob:
            classifier.fit(dataset[i][0], dataset[i][1])
            update_steps_random.append(i) 
        predictions = classifier.predict(dataset[i][0])
        accuracies_random.append(accuracy_score(dataset[i][1], predictions))

    return accuracies_random, update_steps_random


# Initialize storage for results
accuracies_results_random_005 = {}
update_steps_results_random_005 = {}
accuracies_results_random_04 = {}
update_steps_results_random_04 = {}

# Run the "random updates" experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_random_005, update_steps_random_005 = run_random_update_experiment(dataset, classifier, prob_005)
    accuracies_random_04, update_steps_random_04 = run_random_update_experiment(dataset, classifier, prob_04)
    accuracies_results_random_005[key] = accuracies_random_005
    update_steps_results_random_005[key] = update_steps_random_005
    accuracies_results_random_04[key] = accuracies_random_04
    update_steps_results_random_04[key] = update_steps_random_04


# # Evenly Spaced Updates
def compute_even_update_steps(num_batches, target_num_updates):
    """
    Choose exactly target_num_updates update indices from 1..num_batches-1,
    spread as evenly as possible.

    Batch 0 is excluded because it is the initial training batch.
    """
    target_num_updates = int(round(target_num_updates))

    if target_num_updates <= 0:
        return []

    decision_points = np.arange(1, num_batches)

    if target_num_updates >= len(decision_points):
        return decision_points.tolist()

    positions = np.linspace(0, len(decision_points) - 1, target_num_updates)
    update_steps = sorted(
        set(decision_points[np.rint(positions).astype(int)].tolist())
    )

    # Fill any duplicates caused by rounding.
    if len(update_steps) < target_num_updates:
        remaining = [
            step for step in decision_points.tolist()
            if step not in update_steps
        ]
        update_steps = sorted(
            update_steps + remaining[: target_num_updates - len(update_steps)]
        )

    return update_steps


def run_even_update_experiment(
    dataset,
    classifier,
    target_num_updates=average_updates_RL_transfer_005
):
    # Fit the model to the initial dataset
    classifier.fit(dataset[0][0], dataset[0][1])

    accuracies_even = []

    update_steps_even = compute_even_update_steps(
        len(dataset),
        target_num_updates
    )
    update_step_set = set(update_steps_even)

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0][0])
    accuracies_even.append(accuracy_score(dataset[0][1], initial_predictions))

    # Apply updates at the precomputed evenly spaced update indices
    for i in range(1, len(dataset)):
        if i in update_step_set:
            classifier.fit(dataset[i][0], dataset[i][1])

        predictions = classifier.predict(dataset[i][0])
        accuracies_even.append(accuracy_score(dataset[i][1], predictions))

    return accuracies_even, update_steps_even

# Initialize storage for results
accuracies_results_even = {}
update_steps_results_even = {}

# Initialize storage for results
accuracies_results_even_005 = {}
update_steps_results_even_005 = {}
accuracies_results_even_04 = {}
update_steps_results_even_04 = {}

# Run the "evenly spaced updates" experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_even_005, update_steps_even_005 = run_even_update_experiment(dataset, classifier, average_updates_RL_transfer_005)
    accuracies_even_04, update_steps_even_04 = run_even_update_experiment(dataset, classifier, average_updates_RL_transfer_04)
    accuracies_results_even_005[key] = accuracies_even_005
    update_steps_results_even_005[key] = update_steps_even_005
    accuracies_results_even_04[key] = accuracies_even_04
    update_steps_results_even_04[key] = update_steps_even_04


# # DDM

# Define the function to run the DDM process on each dataset
def run_ddm_experiment(datasets, classifier, warning_level=2.0, drift_level=3.0):
    """
    Batch adaptation of DDM (Drift Detection Method).

    The detector is updated once per batch, but the DDM statistic is computed
    from cumulative post-reset instance-level errors:

        p = total_errors / total_instances
        s = sqrt(p * (1 - p) / total_instances)

    Drift is declared when the current p + s exceeds the best historical
    p_min + s_min by drift_level standard deviations. The warning threshold is
    computed for completeness, but only drift triggers model retraining here.

    Evaluation order:
    1. predict/evaluate the current batch with the current model;
    2. compute the current DDM statistic;
    3. test for drift against the previously stored historical minimum;
    4. update the historical minimum only if current statistic is smaller;
    5. if drift is detected, retrain on the current batch for future batches.

    This avoids training on a batch before evaluating that same batch, except
    for batch 0, which is the initial training batch used throughout the
    baseline experiments.
    """
    accuracies_DDM = []
    update_steps_DDM = []

    model = clone(classifier)
    model.fit(datasets[0][0], datasets[0][1])

    total_errors = 0
    total_instances = 0
    p_min = np.inf
    s_min = np.inf

    for i, (X_batch, Y_batch) in enumerate(datasets):
        predictions = model.predict(X_batch)
        accuracy = accuracy_score(Y_batch, predictions)
        accuracies_DDM.append(accuracy)

        batch_errors = int(np.sum(predictions != Y_batch))
        total_errors += batch_errors
        total_instances += len(Y_batch)

        p = total_errors / total_instances
        s = np.sqrt(p * (1 - p) / total_instances)

        # Test for drift BEFORE updating the minimum
        in_warning = (p + s) >= (p_min + warning_level * s_min)
        drift_detected = (i > 0) and ((p + s) >= (p_min + drift_level * s_min))

        # Update the minimum AFTER testing for drift
        if p + s <= p_min + s_min:
            p_min = p
            s_min = s

        if drift_detected:
            model.fit(X_batch, Y_batch)
            update_steps_DDM.append(i)

            # Reset detector after model update. The next batch starts a new
            # post-update monitoring period.
            total_errors = 0
            total_instances = 0
            p_min = np.inf
            s_min = np.inf

    return accuracies_DDM, update_steps_DDM

# Initialize storage for results
accuracies_results_ddm = {}
update_steps_results_ddm = {}

# Run the DDM experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_ddm, update_steps_ddm = run_ddm_experiment(dataset, classifier)
    accuracies_results_ddm[key] = accuracies_ddm
    update_steps_results_ddm[key] = update_steps_ddm

# # FRD

def run_frd_experiment(
    datasets,
    classifier,
    alpha=0.004,
    epsilon=1e-12,
):
    """
    Principled batched adaptation of Four Rates Detector (FRD).

    The original FRD detector monitors four classifier-performance rates:
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)

    In the streaming version, these quantities are updated as labelled
    observations arrive. Here, observations arrive in ordered batches, but the
    observations within each batch are not assumed to have a meaningful order.
    Therefore, each batch contributes one aggregate confusion matrix.

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

    Parameters
    ----------
    datasets : list-like
        Ordered sequence of (X_batch, y_batch) pairs.
    classifier : estimator
        Classifier with fit and predict methods.
    alpha : float, default=0.004
        Overall drift significance level across the monitored rates.
    epsilon : float, default=1e-12
        Numerical tolerance used to avoid division by zero.
    """
    if len(datasets) == 0:
        return [], []

    rate_names = ["TPR", "TNR", "PPV", "NPV"]

    model = clone(classifier)

    accuracies_FRD = []
    update_steps_FRD = []

    # Batch 0 provides the initial model, as in the other updating baselines.
    model.fit(datasets[0][0], datasets[0][1])

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
        predictions = model.predict(X_batch)
        accuracies_FRD.append(accuracy_score(y_batch, predictions))

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
            model.fit(X_batch, y_batch)
            update_steps_FRD.append(i)

            # Reset detector state after adaptation. The drift-triggering batch
            # is used to update the classifier, but not to initialise the new
            # reference period; monitoring restarts from the next batch.
            reference_counts = []
        else:
            # Accept the current batch as belonging to the current post-reset
            # concept and include it in future reference estimates.
            reference_counts.append(current_counts)

    return accuracies_FRD, update_steps_FRD

accuracies_results_frd = {}
update_steps_results_frd = {}

for key, dataset in all_datasets.items():
    accuracies_FRD, update_steps_FRD = run_frd_experiment(
        dataset,
        classifier,
        alpha=0.004,
    )
    accuracies_results_frd[key] = accuracies_FRD
    update_steps_results_frd[key] = update_steps_FRD


# # STEPD

def run_stepd_experiment(
    datasets,
    classifier,
    alpha=0.003,
):
    """
    Batch Statistical Test of Equal Proportions Drift Detection (STEPD).

    STEPD compares the classifier's current batch accuracy with its reference accuracy
    using a one-sided two-sample test for a difference in proportions. The
    detector stores instance-level binary correctness values for post-reset
    batches. At each step, the current batch forms the recent sample, and all
    previous accepted post-reset batches form the reference sample.

    Drift is declared when current accuracy is significantly lower than reference
    accuracy:

        a_ref - a_current > z_{1-alpha} * SE_pooled

    After drift, the classifier is retrained on the current batch and the
    detector state is reset for future batches.
    """
    accuracies_STEPD = []
    update_steps_STEPD = []

    model = clone(classifier)
    model.fit(datasets[0][0], datasets[0][1])

    # Store one numpy array of 0/1 correctness indicators per batch since the
    # last update.
    correctness_batches = []

    for i, (X_batch, Y_batch) in enumerate(datasets):
        predictions = model.predict(X_batch)
        accuracy = accuracy_score(Y_batch, predictions)
        accuracies_STEPD.append(accuracy)

        batch_correctness = (predictions == Y_batch).astype(int)

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
                    * (1 - a_pooled)
                    * (1 / n_ref + 1 / n_recent)
                )

                if standard_error > 0:
                    z_statistic = (a_ref - a_recent) / standard_error
                    threshold = norm.ppf(1 - alpha)
                    drift_detected = z_statistic > threshold

        if drift_detected:
            model.fit(X_batch, Y_batch)
            update_steps_STEPD.append(i)
            correctness_batches = []
        else:
            # Add current batch to reference pool only if drift was not detected
            correctness_batches.append(batch_correctness)

    return accuracies_STEPD, update_steps_STEPD

# Initialize storage for results
accuracies_results_stepd = {}
update_steps_results_stepd = {}

# Run the STEPD experiment for each dataset
for key, dataset in all_datasets.items():
    accuracies_stepd, update_steps_stepd = run_stepd_experiment(dataset, classifier, alpha=0.003)
    accuracies_results_stepd[key] = accuracies_stepd
    update_steps_results_stepd[key] = update_steps_stepd


# # Cumulative Utility for different update penalities

# Initialize dictionaries to store the total utility for each method
total_utility_RL_transfer = {}
total_utility_RL = {}
total_utility_always = {}
total_utility_never = {}
total_utility_random = {}
total_utility_even = {}
total_utility_ddm = {}
total_utility_frd = {}
total_utility_stepd = {}

# calculate the total utility for each method for different update penalties
for penalty in costs:
    # Initialize the cumulative utility for each method
    cumulative_utility_RL_transfer = 0
    cumulative_utility_RL = 0
    cumulative_utility_always = 0
    cumulative_utility_never = 0
    cumulative_utility_random = 0
    cumulative_utility_even = 0
    cumulative_utility_ddm = 0
    cumulative_utility_frd = 0
    cumulative_utility_stepd = 0

    for key in all_datasets:
        if penalty == 0.05:
            # Calculate the cumulative utility for each method
            cumulative_utility_RL_transfer += np.sum(accuracies_results_RL_transfer_005[key]) - penalty * len(update_steps_results_RL_transfer_005[key])
            cumulative_utility_RL += np.sum(accuracies_results_RL_005[key]) - penalty * len(update_steps_results_RL_005[key])
            cumulative_utility_always += np.sum(accuracies_results_always[key]) - penalty * n_timesteps
            cumulative_utility_never += np.sum(accuracies_results_never[key]) - penalty * 0
            cumulative_utility_random += np.sum(accuracies_results_random_005[key]) - penalty * len(update_steps_results_random_005[key])
            cumulative_utility_even += np.sum(accuracies_results_even_005[key]) - penalty * len(update_steps_results_even_005[key])
            cumulative_utility_ddm += np.sum(accuracies_results_ddm[key]) - penalty * len(update_steps_results_ddm[key])
            cumulative_utility_frd += np.sum(accuracies_results_frd[key]) - penalty * len(update_steps_results_frd[key])
            cumulative_utility_stepd += np.sum(accuracies_results_stepd[key]) - penalty * len(update_steps_results_stepd[key])
        else:
            # Calculate the cumulative utility for each method
            cumulative_utility_RL_transfer += np.sum(accuracies_results_RL_transfer_04[key]) - penalty * len(update_steps_results_RL_transfer_04[key])
            cumulative_utility_RL += np.sum(accuracies_results_RL_04[key]) - penalty * len(update_steps_results_RL_04[key])
            cumulative_utility_always += np.sum(accuracies_results_always[key]) - penalty * n_timesteps
            cumulative_utility_never += np.sum(accuracies_results_never[key]) - penalty * 0
            cumulative_utility_random += np.sum(accuracies_results_random_04[key]) - penalty * len(update_steps_results_random_04[key])
            cumulative_utility_even += np.sum(accuracies_results_even_04[key]) - penalty * len(update_steps_results_even_04[key])
            cumulative_utility_ddm += np.sum(accuracies_results_ddm[key]) - penalty * len(update_steps_results_ddm[key])
            cumulative_utility_frd += np.sum(accuracies_results_frd[key]) - penalty * len(update_steps_results_frd[key])
            cumulative_utility_stepd += np.sum(accuracies_results_stepd[key]) - penalty * len(update_steps_results_stepd[key])
    
    # Store the cumulative utility for each method
    total_utility_RL_transfer[penalty] = cumulative_utility_RL_transfer
    total_utility_RL[penalty] = cumulative_utility_RL
    total_utility_always[penalty] = cumulative_utility_always
    total_utility_never[penalty] = cumulative_utility_never
    total_utility_random[penalty] = cumulative_utility_random
    total_utility_even[penalty] = cumulative_utility_even
    total_utility_ddm[penalty] = cumulative_utility_ddm
    total_utility_frd[penalty] = cumulative_utility_frd
    total_utility_stepd[penalty] = cumulative_utility_stepd

# calculate the average utility per episode for each method for different update penalties
average_total_utility_RL_transfer = {penalty: round(total_utility_RL_transfer[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_RL = {penalty: round(total_utility_RL[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_always = {penalty: round(total_utility_always[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_never = {penalty: round(total_utility_never[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_random = {penalty: round(total_utility_random[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_even = {penalty: round(total_utility_even[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_ddm = {penalty: round(total_utility_ddm[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_frd = {penalty: round(total_utility_frd[penalty] / len(all_datasets),1) for penalty in costs}
average_total_utility_stepd = {penalty: round(total_utility_stepd[penalty] / len(all_datasets),1) for penalty in costs}

# Create a DataFrame using the update penalties as the index and the methods as columns
utilities = pd.DataFrame({
    'RL (Transfer)': average_total_utility_RL_transfer,
    'RL': average_total_utility_RL,
    'Always': average_total_utility_always,
    'Never': average_total_utility_never,
    'Random': average_total_utility_random,
    'Even': average_total_utility_even,
    'DDM': average_total_utility_ddm,
    'FRD': average_total_utility_frd,
    'STEPD': average_total_utility_stepd
})

# Display the DataFrame
utilities.index.name = 'Update Penalty'
print(utilities.T)


# Helper dictionaries for easier access
accuracy_methods = {
    'AMUSE Transfer (0.05)': accuracies_results_RL_transfer_005,
    'AMUSE Transfer (0.4)': accuracies_results_RL_transfer_04,
    'AMUSE (0.05)': accuracies_results_RL_005,
    'AMUSE (0.4)': accuracies_results_RL_04,
    'Always Update': accuracies_results_always,
    'Never Update': accuracies_results_never,
    'Random (0.05)': accuracies_results_random_005,
    'Random (0.4)': accuracies_results_random_04,
    'Even (0.05)': accuracies_results_even_005,
    'Even (0.4)': accuracies_results_even_04,
    'DDM': accuracies_results_ddm,
    'FRD': accuracies_results_frd,
    'STEPD': accuracies_results_stepd
}

# Collect average accuracies for each method across all datasets
data = {'Method': [], 'Average Accuracy': []}

for method, results in accuracy_methods.items():
    for key in all_datasets.keys():
        avg_accuracy = np.mean(results[key])
        data['Method'].append(method)
        data['Average Accuracy'].append(avg_accuracy)

# Create a Pandas DataFrame for the collected data
df = pd.DataFrame(data)

output_dir = 'Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Save the DataFrame to a CSV file
df.to_csv('Outputs/average_accuracies.csv', index=False)

#average_updates is the average length of the update steps for each method
average_updates = {
    'AMUSE Transfer (0.05)': np.mean([len(update_steps_results_RL_transfer_005[key]) for key in update_steps_results_RL_transfer_005]),
    'AMUSE Transfer (0.4)': np.mean([len(update_steps_results_RL_transfer_04[key]) for key in update_steps_results_RL_transfer_04]),
    'AMUSE (0.05)': np.mean([len(update_steps_results_RL_005[key]) for key in update_steps_results_RL_005]),
    'AMUSE (0.4)': np.mean([len(update_steps_results_RL_04[key]) for key in update_steps_results_RL_04]),
    'DDM': np.mean([len(update_steps_results_ddm[key]) for key in update_steps_results_ddm]),
    'FRD': np.mean([len(update_steps_results_frd[key]) for key in update_steps_results_frd]),
    'STEPD': np.mean([len(update_steps_results_stepd[key]) for key in update_steps_results_stepd]),
    'Random (0.05)': np.mean([len(update_steps_results_random_005[key]) for key in update_steps_results_random_005]),
    'Random (0.4)': np.mean([len(update_steps_results_random_04[key]) for key in update_steps_results_random_04]),
    'Even (0.05)': np.mean([len(update_steps_results_even_005[key]) for key in update_steps_results_even_005]),
    'Even (0.4)': np.mean([len(update_steps_results_even_04[key]) for key in update_steps_results_even_04]),
    'Always Update': n_timesteps,
    'Never Update': 0
}


# print the average number of updates for each method
print(average_updates)


# save all accuracies and update_steps into a dictionary
all_results = {
    'accuracies': {
        'AMUSE Transfer (0.05)': accuracies_results_RL_transfer_005,
        'AMUSE Transfer (0.4)': accuracies_results_RL_transfer_04,
        'AMUSE (0.05)': accuracies_results_RL_005,
        'AMUSE (0.4)': accuracies_results_RL_04,
        'Always Update': accuracies_results_always,
        'Never Update': accuracies_results_never,
        'Random (0.05)': accuracies_results_random_005,
        'Random (0.4)': accuracies_results_random_04,
        'Even (0.05)': accuracies_results_even_005,
        'Even (0.4)': accuracies_results_even_04,
        'DDM': accuracies_results_ddm,
        'FRD': accuracies_results_frd,
        'STEPD': accuracies_results_stepd
    },
    'update_steps': {
        'AMUSE Transfer (0.05)': update_steps_results_RL_transfer_005,
        'AMUSE Transfer (0.4)': update_steps_results_RL_transfer_04,
        'AMUSE (0.05)': update_steps_results_RL_005,
        'AMUSE (0.4)': update_steps_results_RL_04,
        'Always Update': {},
        'Never Update': {},
        'Random (0.05)': update_steps_results_random_005,
        'Random (0.4)': update_steps_results_random_04,
        'Even (0.05)': update_steps_results_even_005,
        'Even (0.4)': update_steps_results_even_04,
        'DDM': update_steps_results_ddm,
        'FRD': update_steps_results_frd,
        'STEPD': update_steps_results_stepd
    },
    'average_total_utility': {
        'AMUSE Transfer (0.05)': total_utility_RL_transfer,
        'AMUSE': total_utility_RL,
        'Always Update': total_utility_always,
        'Never Update': total_utility_never,
        'Random': total_utility_random,
        'Even': total_utility_even,
        'DDM': total_utility_ddm,
        'FRD': total_utility_frd,
        'STEPD': total_utility_stepd
    },
    'average_updates': average_updates
}

# Save the dictionary to a file
np.save('Outputs/results.npy', all_results)

# save the utilities dataframe to a csv file
utilities.T.to_csv('Outputs/utilities.csv')

# save the average number of updates for each method to a csv file
pd.DataFrame(average_updates, index=[0]).T.to_csv('Outputs/average_updates.csv')