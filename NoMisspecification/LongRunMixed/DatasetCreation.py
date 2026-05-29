import numpy as np
import os

# Generate initial dataset
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

n_samples = 1000
n_features = 5
n_timesteps = 50000

# Generate dataset
X = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

initial_real_coefficients = np.random.uniform(-0.75, 0.75, size=n_features)

# Calculate logits
logits = np.dot(X, initial_real_coefficients)

# Apply sigmoid function to get probabilities
probs = 1 / (1 + np.exp(-logits))

# Generate binary target variable
Y = np.random.binomial(1, probs)

# Store the initial dataset
initial_dataset = (X, Y)


# Function to generate drifted datasets with sudden drift
def generate_drifted_datasets(n_timesteps, n_features, X, initial_real_coefficients, probs, seed, sudden_drift_prob=0.01):
    np.random.seed(seed)
    
    dataset = []
    coefficients = []
    probs_list = []

    # Initialize coefficients and probabilities
    coefficients.append(initial_real_coefficients)
    probs_list.append(probs)

    # Store initial dataset
    Y = np.random.binomial(1, probs)
    dataset.append((X, Y))

    # Generate new datasets with sudden drift
    for episode in range(1,n_timesteps):
        if np.random.rand() < sudden_drift_prob:
            # Sudden drift occurs: change coefficients randomly
            new_coefficients = np.random.uniform(low=-1, high=1, size=n_features)
            coefficients.append(new_coefficients)
        else:
            drift = np.random.normal(loc=0, scale=0.05, size=n_features)
            # Apply drift and clip within bounds
            coefficients.append(np.clip(coefficients[-1] + drift, -1, 1))
        
        # Generate dataset
        X = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

        # Update probabilities and generate new data
        logits = np.dot(X, coefficients[-1])
        probs = 1 / (1 + np.exp(-logits))
        probs_list.append(probs)
        Y = np.random.binomial(1, probs)
        dataset.append((X, Y))
    
    return dataset


# Generate set of drifted datasets with different seeds
seeds = np.random.randint(0, 1000, 1)
datasets = {}

for i, seed in enumerate(seeds):
    datasets[i] = generate_drifted_datasets(n_timesteps, n_features, X, initial_real_coefficients, probs, seed)


# Save the datasets, initial_real_coefficients, probs
if not os.path.exists("Datasets"):
    os.makedirs("Datasets")

np.savez("Datasets/initial_dataset.npz", X=X, Y=Y)
np.save("Datasets/drifted_datasets.npy", datasets, allow_pickle=True)
np.save("Datasets/initial_real_coefficients.npy", initial_real_coefficients)
np.save("Datasets/probs.npy", probs)

