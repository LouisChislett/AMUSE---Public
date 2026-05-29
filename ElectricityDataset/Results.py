import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO
import os
from scipy.stats import norm
from sklearn.base import clone
import pickle

from RealEnvironment import ModelUpdatingEnv

# set the seed
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)


# Load the weekly batches list from the .pkl file
with open("Datasets/drifted_datasets.pkl", "rb") as f:
    drifted_datasets = pickle.load(f)


len(drifted_datasets)
n_timesteps = len(drifted_datasets)

# Define the update penalties to test
costs = [0.05, 0.4]

# # Reinforcement Learning without Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_005 = {}
update_steps_results_RL_005 = {}

# Run the model for each set of drifted datasets
# Load the pre-trained model
model_005 = PPO.load(f'Models/best_model_0.05.zip')


# Set up the environment for each set of drifted datasets
np.random.seed(144)
env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.01, n_timesteps=n_timesteps)
    
obs = env.reset()
done = False

# Run the environment until done
while not done:
    action, _ = model_005.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
# Collect the accuracy log and update steps for this set
accuracy_log_005 = env.get_accuracy_log()
update_steps_005 = env.get_update_steps()

#remove the first element of the update steps
update_steps_005 = update_steps_005[1:]

# Store the results in the dictionary with the key
accuracies_results_RL_005 = accuracy_log_005
update_steps_results_RL_005 = update_steps_005

prob_005 = len(update_steps_results_RL_005) / n_timesteps
average_updates_RL_005 = len(update_steps_results_RL_005)


# # Reinforcement Learning without Transfer Learning

# Initialize a dictionary to store the accuracy logs and update steps for each set
accuracies_results_RL_04 = {}
update_steps_results_RL_04 = {}

# Run the model for each set of drifted datasets
# Load the pre-trained model
model_04 = PPO.load(f'Models/best_model_0.4.zip')


# Set up the environment for each set of drifted datasets
np.random.seed(144)
env = ModelUpdatingEnv(drifted_datasets, update_penalty=0.01, n_timesteps=n_timesteps)
    
obs = env.reset()
done = False

# Run the environment until done
while not done:
    action, _ = model_04.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
# Collect the accuracy log and update steps for this set
accuracy_log_04 = env.get_accuracy_log()
update_steps_04 = env.get_update_steps()

#remove the first element of the update steps
update_steps_04 = update_steps_04[1:]

# Store the results in the dictionary with the key
accuracies_results_RL_04 = accuracy_log_04
update_steps_results_RL_04 = update_steps_04

prob_04 = len(update_steps_results_RL_04) / n_timesteps
average_updates_RL_04 = len(update_steps_results_RL_04)

print(prob_005, prob_04, average_updates_RL_005, average_updates_RL_04)

# define the classifier for other experiments
classifier = LogisticRegression(random_state=144, max_iter=1000)

# # Always Update

# Define the function to run the "always update" process on each dataset
def run_always_update_experiment(dataset, classifier):
    # Fit the model to the initial dataset
    classifier.fit(dataset[0].drop(columns=['class', 'date', 'day']), dataset[0]['class'])

    # Initialize a way of storing the accuracies
    accuracies_always = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0].drop(columns=['class', 'date', 'day']))
    accuracies_always.append(accuracy_score(dataset[0]['class'], initial_predictions))

    # Cycle through the datasets, applying updates and recording accuracies
    for i in range(1, len(dataset)):
        classifier.fit(dataset[i].drop(columns=['class', 'date', 'day']), dataset[i]['class'])
        predictions = classifier.predict(dataset[i].drop(columns=['class', 'date', 'day']))
        accuracies_always.append(accuracy_score(dataset[i]['class'], predictions))

    return accuracies_always

accuracies_always = run_always_update_experiment(drifted_datasets, classifier)
accuracies_results_always = accuracies_always


# # Never Update

# Define the function to run the "never update" process on each dataset
def run_never_update_experiment(dataset, classifier):

    # Fit the model to the initial dataset
    classifier.fit(dataset[0].drop(columns=['class', 'date', 'day']), dataset[0]['class'])

    # Initialize a way of storing the accuracies
    accuracies_never = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0].drop(columns=['class', 'date', 'day']))
    accuracies_never.append(accuracy_score(dataset[0]['class'], initial_predictions))

    # Cycle through the datasets without applying updates, recording accuracies
    for i in range(1, len(dataset)):
        predictions = classifier.predict(dataset[i].drop(columns=['class', 'date', 'day']))
        accuracies_never.append(accuracy_score(dataset[i]['class'], predictions))

    return accuracies_never

accuracies_never = run_never_update_experiment(drifted_datasets, classifier)
accuracies_results_never = accuracies_never


# # Random Updates
np.random.seed(144)
# Define the function to run the "random updates" process on each dataset
def run_random_update_experiment(dataset, classifier, update_prob=prob_005):

    # Fit the model to the initial dataset
    classifier.fit(dataset[0].drop(columns=['class', 'date', 'day']), dataset[0]['class'])

    # Initialize a way of storing the accuracies
    accuracies_random = []
    cels_random = []
    update_steps_random = []

    # Compute initial accuracy
    initial_predictions = classifier.predict(dataset[0].drop(columns=['class', 'date', 'day']))
    accuracies_random.append(accuracy_score(dataset[0]['class'], initial_predictions))

    # Cycle through the datasets, applying random updates and recording accuracies
    for i in range(1, len(dataset)):
        if np.random.uniform() < update_prob:
            classifier.fit(dataset[i].drop(columns=['class', 'date', 'day']), dataset[i]['class'])
            update_steps_random.append(i) 
        predictions = classifier.predict(dataset[i].drop(columns=['class', 'date', 'day']))
        accuracies_random.append(accuracy_score(dataset[i]['class'], predictions))

    return accuracies_random, update_steps_random

# Run the "random updates" experiment for each dataset
accuracies_results_random_005, update_steps_results_random_005 = run_random_update_experiment(drifted_datasets, classifier, prob_005)

# Run the "random updates" experiment for each dataset
accuracies_results_random_04, update_steps_results_random_04 = run_random_update_experiment(drifted_datasets, classifier, prob_04)

# # Evenly spaced updates

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


def run_even_update_experiment(dataset, classifier, target_num_updates):
    # Fit the model to the initial dataset
    classifier.fit(
        dataset[0].drop(columns=['class', 'date', 'day']),
        dataset[0]['class']
    )

    accuracies_even = []

    update_steps_even = compute_even_update_steps(
        len(dataset),
        target_num_updates
    )
    update_step_set = set(update_steps_even)

    # Compute initial accuracy
    initial_predictions = classifier.predict(
        dataset[0].drop(columns=['class', 'date', 'day'])
    )
    accuracies_even.append(
        accuracy_score(dataset[0]['class'], initial_predictions)
    )

    # Apply updates at the precomputed evenly spaced update indices
    for i in range(1, len(dataset)):
        X_batch = dataset[i].drop(columns=['class', 'date', 'day'])
        y_batch = dataset[i]['class']

        if i in update_step_set:
            classifier.fit(X_batch, y_batch)

        predictions = classifier.predict(X_batch)
        accuracies_even.append(accuracy_score(y_batch, predictions))

    return accuracies_even, update_steps_even

accuracies_results_even_005, update_steps_results_even_005 = run_even_update_experiment(
    drifted_datasets,
    classifier,
    target_num_updates=len(update_steps_results_RL_005)
)

accuracies_results_even_04, update_steps_results_even_04 = run_even_update_experiment(
    drifted_datasets,
    classifier,
    target_num_updates=len(update_steps_results_RL_04)
)


# # DDM

# Define the function to run the DDM process on the electricity dataset
def run_ddm_experiment(datasets, classifier, warning_level=2.0, drift_level=3.0):
    """
    Batch adaptation of DDM (Drift Detection Method) for the electricity data.

    Evaluation order:
    1. fit the initial classifier on batch 0;
    2. predict/evaluate each batch with the current model;
    3. compute the current cumulative DDM statistic since the most recent reset;
    4. test for drift against the previously stored historical minimum;
    5. update the historical minimum only after the drift test;
    6. if drift is detected, retrain on the current batch for future batches
       and reset the detector state.

    Batch 0 is used as the initial training batch, as in the other baselines.
    """
    if len(datasets) == 0:
        return [], []

    accuracies_DDM = []
    update_steps_DDM = []

    model = clone(classifier)

    X_0 = datasets[0].drop(columns=['class', 'date', 'day'])
    y_0 = datasets[0]['class']
    model.fit(X_0, y_0)

    total_errors = 0
    total_instances = 0
    p_min = np.inf
    s_min = np.inf

    for i, dataset in enumerate(datasets):
        X_batch = dataset.drop(columns=['class', 'date', 'day'])
        y_batch = dataset['class']

        predictions = model.predict(X_batch)
        accuracy = accuracy_score(y_batch, predictions)
        accuracies_DDM.append(accuracy)

        batch_errors = int(np.sum(predictions != y_batch))
        total_errors += batch_errors
        total_instances += len(y_batch)

        p = total_errors / total_instances
        s = np.sqrt(p * (1 - p) / total_instances)

        # Test for drift against the previous historical minimum before
        # allowing the current batch to update that minimum.
        in_warning = (p + s) >= (p_min + warning_level * s_min)
        drift_detected = (i > 0) and ((p + s) >= (p_min + drift_level * s_min))

        if p + s <= p_min + s_min:
            p_min = p
            s_min = s

        if drift_detected:
            model.fit(X_batch, y_batch)
            update_steps_DDM.append(i)

            # Reset detector after model update. The next batch starts a new
            # post-update monitoring period.
            total_errors = 0
            total_instances = 0
            p_min = np.inf
            s_min = np.inf

    return accuracies_DDM, update_steps_DDM

# Run the DDM experiment for the electricity dataset
accuracies_results_ddm, update_steps_results_ddm = run_ddm_experiment(drifted_datasets, classifier)


# # FRD

def run_frd_experiment(
    datasets,
    classifier,
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

    Each electricity batch contributes one aggregate confusion matrix. For each
    rate, the current batch rate is compared with the denominator-weighted
    post-reset reference rate using a one-sided two-proportion z-test. Positive
    z-statistics indicate deterioration.

    Drift is declared if any defined and testable rate is significantly lower
    than its reference value, using a Bonferroni correction across the rates
    that are testable for the current batch.

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
    if len(datasets) == 0:
        return [], []

    rate_names = ["TPR", "TNR", "PPV", "NPV"]

    model = clone(classifier)

    accuracies_FRD = []
    update_steps_FRD = []

    # Batch 0 provides the initial model, as in the other updating baselines.
    X_0 = datasets[0].drop(columns=['class', 'date', 'day'])
    y_0 = datasets[0]['class']
    model.fit(X_0, y_0)

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

    for i, dataset in enumerate(datasets):
        X_batch = dataset.drop(columns=['class', 'date', 'day'])
        y_batch = dataset['class']

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

# Run the FRD experiment for the electricity dataset
accuracies_results_frd, update_steps_results_frd = run_frd_experiment(
    drifted_datasets,
    classifier,
    alpha=0.004,
)


# # STEPD

def run_stepd_experiment(
    datasets,
    classifier,
    alpha=0.003,
):
    """
    Batch Statistical Test of Equal Proportions Drift Detection (STEPD).

    STEPD compares the classifier's current batch accuracy with its reference
    accuracy using a one-sided two-sample test for a difference in proportions.
    The detector stores instance-level binary correctness values for accepted
    post-reset batches. At each step, the current batch is the recent sample,
    and all previous accepted post-reset batches are the reference sample.

    Drift is declared when current accuracy is significantly lower than
    reference accuracy. After drift, the classifier is retrained on the current
    batch and the detector state is reset for future batches.
    """
    if len(datasets) == 0:
        return [], []

    accuracies_STEPD = []
    update_steps_STEPD = []

    model = clone(classifier)

    X_0 = datasets[0].drop(columns=['class', 'date', 'day'])
    y_0 = datasets[0]['class']
    model.fit(X_0, y_0)

    # Store one numpy array of 0/1 correctness indicators per accepted batch
    # since the last update.
    correctness_batches = []

    for i, dataset in enumerate(datasets):
        X_batch = dataset.drop(columns=['class', 'date', 'day'])
        y_batch = dataset['class']

        predictions = model.predict(X_batch)
        accuracy = accuracy_score(y_batch, predictions)
        accuracies_STEPD.append(accuracy)

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
                    * (1 - a_pooled)
                    * (1 / n_ref + 1 / n_recent)
                )

                if standard_error > 0:
                    z_statistic = (a_ref - a_recent) / standard_error
                    threshold = norm.ppf(1 - alpha)
                    drift_detected = z_statistic > threshold

        if drift_detected:
            model.fit(X_batch, y_batch)
            update_steps_STEPD.append(i)
            correctness_batches = []
        else:
            # Add current batch to reference pool only if drift was not detected.
            correctness_batches.append(batch_correctness)

    return accuracies_STEPD, update_steps_STEPD

# Run the STEPD experiment for the electricity dataset
accuracies_results_stepd, update_steps_results_stepd = run_stepd_experiment(
    drifted_datasets,
    classifier,
    alpha=0.003,
)

# # Cumulative Utility for different update penalities

# Initialize dictionaries to store the total utility for each method
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
    cumulative_utility_RL = 0
    cumulative_utility_always = 0
    cumulative_utility_never = 0
    cumulative_utility_random = 0
    cumulative_utility_even = 0
    cumulative_utility_ddm = 0
    cumulative_utility_frd = 0
    cumulative_utility_stepd = 0

    if penalty == 0.05:
        # Calculate the cumulative utility for each method
        cumulative_utility_RL += np.sum(accuracies_results_RL_005) - penalty * len(update_steps_results_RL_005)
        cumulative_utility_always += np.sum(accuracies_results_always) - penalty * n_timesteps
        cumulative_utility_never += np.sum(accuracies_results_never) - penalty * 0
        cumulative_utility_random += np.sum(accuracies_results_random_005) - penalty * len(update_steps_results_random_005)
        cumulative_utility_even += np.sum(accuracies_results_even_005) - penalty * len(update_steps_results_even_005)
        cumulative_utility_ddm += np.sum(accuracies_results_ddm) - penalty * len(update_steps_results_ddm)
        cumulative_utility_frd += np.sum(accuracies_results_frd) - penalty * len(update_steps_results_frd)
        cumulative_utility_stepd += np.sum(accuracies_results_stepd) - penalty * len(update_steps_results_stepd)
    else:
        # Calculate the cumulative utility for each method
        cumulative_utility_RL += np.sum(accuracies_results_RL_04) - penalty * len(update_steps_results_RL_04)
        cumulative_utility_always += np.sum(accuracies_results_always) - penalty * n_timesteps
        cumulative_utility_never += np.sum(accuracies_results_never) - penalty * 0
        cumulative_utility_random += np.sum(accuracies_results_random_04) - penalty * len(update_steps_results_random_04)
        cumulative_utility_even += np.sum(accuracies_results_even_04) - penalty * len(update_steps_results_even_04)
        cumulative_utility_ddm += np.sum(accuracies_results_ddm) - penalty * len(update_steps_results_ddm)
        cumulative_utility_frd += np.sum(accuracies_results_frd) - penalty * len(update_steps_results_frd)
        cumulative_utility_stepd += np.sum(accuracies_results_stepd) - penalty * len(update_steps_results_stepd)

    # Store the cumulative utility for each method
    total_utility_RL[penalty] = cumulative_utility_RL
    total_utility_always[penalty] = cumulative_utility_always
    total_utility_never[penalty] = cumulative_utility_never
    total_utility_random[penalty] = cumulative_utility_random
    total_utility_even[penalty] = cumulative_utility_even
    total_utility_ddm[penalty] = cumulative_utility_ddm
    total_utility_frd[penalty] = cumulative_utility_frd
    total_utility_stepd[penalty] = cumulative_utility_stepd

# Create a DataFrame using the update penalties as the index and the methods as columns
utilities = pd.DataFrame({
    'RL': total_utility_RL,
    'Always': total_utility_always,
    'Never': total_utility_never,
    'Random': total_utility_random,
    'Even': total_utility_even,
    'DDM': total_utility_ddm,
    'FRD': total_utility_frd,
    'STEPD': total_utility_stepd
})

# Display the DataFrame
utilities.index.name = 'Update Penalty'
print(utilities.T)

# Helper dictionaries for easier access
accuracy_methods = {
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

# Collect average accuracies for each method
data = {'Method': [], 'Average Accuracy': []}

for method, results in accuracy_methods.items():
    avg_accuracy = np.mean(results)
    data['Method'].append(method)
    data['Average Accuracy'].append(avg_accuracy)

# Create a Pandas DataFrame for the collected data
df = pd.DataFrame(data)

output_dir = 'Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the DataFrame to a CSV file
df.to_csv('Outputs/average_accuracies.csv', index=False)

# average_updates is the number of update steps for each method
average_updates = {
    'AMUSE (0.05)': len(update_steps_results_RL_005),
    'AMUSE (0.4)': len(update_steps_results_RL_04),
    'DDM': len(update_steps_results_ddm),
    'FRD': len(update_steps_results_frd),
    'STEPD': len(update_steps_results_stepd),
    'Random (0.05)': len(update_steps_results_random_005),
    'Random (0.4)': len(update_steps_results_random_04),
    'Even (0.05)': len(update_steps_results_even_005),
    'Even (0.4)': len(update_steps_results_even_04),
    'Always Update': n_timesteps,
    'Never Update': 0
}

# print the average number of updates for each method
print(average_updates)

# save all accuracies and update_steps into a dictionary
all_results = {
    'accuracies': {
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
