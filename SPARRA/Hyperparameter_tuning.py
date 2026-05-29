import numpy as np
from stable_baselines3 import PPO
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

from ModelUpdatingEnv import ModelUpdatingEnv

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

# Load and process data
csv_path = r"PATH"
initial_dataset = pd.read_csv(csv_path)

# Build Y (Target) and X (age + decile + sexM)
required_cols = ["target", "age", "decile", "sexM"]
missing = [c for c in required_cols if c not in initial_dataset.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Ensure age is numeric
initial_dataset["age"] = pd.to_numeric(initial_dataset["age"], errors="coerce")
initial_dataset["decile"] = initial_dataset["decile"].astype("category")

# Ensure sexM is numeric
initial_dataset["sexM"] = pd.to_numeric(initial_dataset["sexM"], errors="coerce")

# Drop rows with missing values
initial_dataset = initial_dataset.dropna(subset=required_cols).copy()

# Sample 1000 rows for faster evaluation
initial_dataset = initial_dataset.sample(n=min(1000, len(initial_dataset)), random_state=144).reset_index(drop=True)
print(f"Using {len(initial_dataset)} samples for hyperparameter tuning")

# Target vector
Y = initial_dataset["target"]

# Convert target to binary
if Y.dtype == "object" or str(Y.dtype).startswith("category"):
    Y = Y.astype(str).str.strip().str.lower()
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
    if set(Y.unique()).issubset(set(mapping.keys())):
        Y = Y.map(mapping)
    else:
        codes, uniques = pd.factorize(Y)
        if len(uniques) != 2:
            raise ValueError(f"Target must be binary; found classes: {list(uniques)}")
        Y = pd.Series(codes, index=initial_dataset.index)

# Create feature matrix
X = pd.DataFrame({
    "age": initial_dataset["age"].astype(float),
    "sexM": pd.to_numeric(initial_dataset["sexM"], errors="coerce").astype(int)
})
decile_dummies = pd.get_dummies(initial_dataset["decile"], prefix="decile", drop_first=True)
X = pd.concat([X, decile_dummies], axis=1)
feature_names = X.columns.tolist()

# Validate classes
unique_classes = np.unique(Y)
if len(unique_classes) != 2:
    raise ValueError(f"Target must be binary. Found {len(unique_classes)} class(es)")

# Convert to numpy arrays
Y = Y.values
X = X.values

# Fit initial model
logit_model = LogisticRegression(max_iter=2000)
logit_model.fit(X, Y)

initial_real_coefficients = logit_model.coef_[0]
probs = logit_model.predict_proba(X)[:, 1]

n_features = X.shape[1]
n_timesteps = 100

# Load trained models and run evaluation
print("\n" + "="*60)
print("Running Hyperparameter Tuning Evaluation")
print("="*60)

# Dictionary to store the loaded models
models = {}
update_penalties = [0.01, 0.02, 0.03, 0.04, 0.05]
n_episodes = 10  # Number of episodes to run

# Load the trained models
for penalty in update_penalties:
    try:
        models[penalty] = PPO.load(f'Models/penalty_{penalty}.zip')
        print(f"✓ Loaded model for penalty={penalty}")
    except Exception as e:
        print(f"✗ Failed to load model for penalty={penalty}: {e}")
        continue

if not models:
    print("\n✗ No models loaded. Exiting.")
    exit(1)

# To store AUC and log-loss logs and update steps for each episode
episode_auc_logs = {penalty: [] for penalty in update_penalties}
episode_cel_logs = {penalty: [] for penalty in update_penalties}
episode_update_steps = {penalty: [] for penalty in update_penalties}

print(f"\nRunning {n_episodes} evaluation episodes per model...")
# Run for multiple episodes and collect the results
for penalty, model in models.items():
    for episode in range(n_episodes):
        np.random.seed(144 + episode)
        env = ModelUpdatingEnv(X, Y, initial_real_coefficients, probs, penalty, n_features, n_timesteps)
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        # Collect the AUC and log-loss logs and update steps for this episode
        episode_auc_logs[penalty].append(np.array(env.episode_auc_log))
        episode_cel_logs[penalty].append(np.array(env.episode_cel_log))
        episode_update_steps[penalty].append(env.get_update_steps())

# Combine logs across episodes for each penalty
total_auc_log = {}
total_cel_log = {}
total_update_steps = {}

for penalty in update_penalties:
    # Concatenate AUC logs across episodes for the current penalty
    total_auc_log[penalty] = np.concatenate(episode_auc_logs[penalty])
    
    # Concatenate log-loss logs across episodes for the current penalty
    total_cel_log[penalty] = np.concatenate(episode_cel_logs[penalty])
    
    # Concatenate update steps across episodes for the current penalty
    total_update_steps[penalty] = np.concatenate(episode_update_steps[penalty])

# Calculate cumulative utility for each model based on AUC gain vs update cost
costs = [0.05, 0.4]
cumulative_utilities = {}

print("\nCalculating cumulative utilities...")
for cost in costs:
    cumulative_utility = {}
    for penalty in update_penalties:
        # For each penalty, calculate cumulative utility: AUC gain - cost * number of updates
        auc_gain = np.sum(total_auc_log[penalty])
        num_updates = len(total_update_steps[penalty])
        cumulative_utility[penalty] = auc_gain - cost * num_updates
    cumulative_utilities[cost] = cumulative_utility

# Make a table of cumulative utilities for each model for different costs
df = pd.DataFrame(cumulative_utilities)
df.index.name = "Update Penalty"
df.columns.name = "Cost"
print("\nCumulative Utility Analysis:")
print(df)

# Save all cumulative utilities for each model/update penalty and cost
os.makedirs("Outputs", exist_ok=True)
utilities_csv_path = os.path.join("Outputs", "hyperparameter_tuning_utilities.csv")
df.to_csv(utilities_csv_path)
print(f"Saved cumulative utilities to {utilities_csv_path}")

# Pick the best model for each cost
best_models = {}
best_penalties = {}
for cost in costs:
    best_penalty = max(cumulative_utilities[cost], key=cumulative_utilities[cost].get)
    best_penalties[cost] = best_penalty
    best_models[cost] = models[best_penalty]
    print(f"\nBest penalty for cost={cost}: {best_penalty} (utility: {cumulative_utilities[cost][best_penalty]:.4f})")

# Save the best penalties to a file
with open("Models/best_penalties.txt", "w") as f:
    for cost, penalty in best_penalties.items():
        f.write(f"{cost}: {penalty}\n")
print("\n✓ Saved best penalties to Models/best_penalties.txt")

# Save the best model for each cost
for cost, model in best_models.items():
    model.save(f"Models/best_model_{cost}.zip")
    print(f"✓ Saved best model for cost={cost} to Models/best_model_{cost}.zip")

# Save the hyperparameters for the best models
for cost, model in best_models.items():
    hyperparameters = {
        "Learning rate": model.learning_rate,
        "Batch size": model.batch_size,
        "Number of steps": model.n_steps,
        "Gamma": model.gamma,
        "GAE lambda": model.gae_lambda,
        "Clip range": model.clip_range,
        "Max gradient norm": model.max_grad_norm,
        "Entropy coefficient": model.ent_coef,
        "Value function coefficient": model.vf_coef,
        "Number of epochs": model.n_epochs,
        "Update penalty": best_penalties[cost]
    }

    # Save the hyperparameters to a file
    with open(f"Models/hyperparameters_{cost}.txt", "w") as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Saved hyperparameters for cost={cost} to Models/hyperparameters_{cost}.txt")

