import numpy as np
from stable_baselines3 import PPO
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ModelUpdatingEnv import ModelUpdatingEnv

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

# Dictionary to store the loaded models
models = {}
update_penalties = [0.01, 0.02, 0.03, 0.04, 0.05] # List of update penalties
n_episodes = 10 # Number of episodes to run

# Load the data
data = np.load("Datasets/initial_dataset.npz")
X = data['X']
Y = data['Y']

# We will estimate the initial_real_coefficients and the initial_probs using a logistic regression model
model = LogisticRegression()
model.fit(X, Y)
initial_real_coefficients = model.coef_[0]
probs = model.predict_proba(X)[:, 1]

n_features = X.shape[1]
n_timesteps = 100

# Loop through the penalties and load each model from the Models folder
for penalty in update_penalties:
    models[penalty] = PPO.load(f'Models/penalty_{penalty}.zip')

# To store accuracy logs and update steps for each episode
episode_accuracy_logs = {penalty: [] for penalty in update_penalties}
episode_update_steps = {penalty: [] for penalty in update_penalties}

# Run for multiple episodes and collect the results
for penalty, model in models.items():
    for episode in range(n_episodes):
        np.random.seed(144+episode)
        env = ModelUpdatingEnv(X, Y, initial_real_coefficients, probs, penalty, n_features, n_timesteps)
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        # Collect the accuracy logs and update steps for this episode
        episode_accuracy_logs[penalty].append(np.array(env.get_accuracy_log()))
        episode_update_steps[penalty].append(env.get_update_steps())

# Combine logs across episodes for each penalty
total_accuracy_log = {}
total_update_steps = {}

for penalty in update_penalties:
    # Concatenate accuracy logs across episodes for the current penalty
    total_accuracy_log[penalty] = np.concatenate(episode_accuracy_logs[penalty])
    
    # Concatenate update steps across episodes for the current penalty
    total_update_steps[penalty] = np.concatenate(episode_update_steps[penalty])

# Now that we have logs for all episodes, calculate cumulative utility for each model and each cost
# Now that we have logs for all episodes, calculate cumulative utility for each model and each cost
costs = [0.05, 0.4]
cumulative_utilities = {}

for cost in costs:
    cumulative_utility = {}
    for penalty in update_penalties:
        # For each penalty, calculate cumulative utility
        cumulative_utility[penalty] = np.sum(total_accuracy_log[penalty]) - cost * len(total_update_steps[penalty])
    cumulative_utilities[cost] = cumulative_utility

# Make a table of cumulative utilities for each model for different costs
df = pd.DataFrame(cumulative_utilities)

# Save cumulative utilities to the Outputs folder
df.to_csv("Outputs/hyperparameter_tuning_utilities.csv")

best_models = {}
best_penalties = {}
for cost in costs:
    best_penalty = max(cumulative_utilities[cost], key=cumulative_utilities[cost].get)
    best_penalties[cost] = best_penalty
    best_models[cost] = models[best_penalty]
    # save the best penalty for each cost
    with open(f'Models/best_penalty_{cost}.txt', 'w') as f:
        f.write(f"{best_penalty}")

# Save the best model for each cost
for cost, model in best_models.items():
    model.save(f"Models/best_model_{cost}.zip")

# Save the best penalties to a file
with open("Models/best_penalties.txt", "w") as f:
    for cost, penalty in best_penalties.items():
        f.write(f"{cost}: {penalty}\n")

# Save the hyperparameters for the best models
for cost, model in best_models.items():
    hyperparameters = {
        "Learning rate": model.learning_rate,
        "Batch size": model.batch_size,     #M
        "Number of steps": model.n_steps,       #T
        "Gamma": model.gamma,
        "GAE lambda": model.gae_lambda,
        "VF coefficient": model.vf_coef,
        "Clip range": model.clip_range,
        "Max gradient norm": model.max_grad_norm,
        "Entropy coefficient": model.ent_coef,
        "Value function coefficient": model.vf_coef,
        "Number of epochs": model.n_epochs,      #K
        "Update penalty": best_penalties[cost]
    }

    # Save the hyperparameters to a file
    with open(f'Models/hyperparameters_{cost}.txt', 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")