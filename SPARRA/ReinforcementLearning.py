import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pandas as pd
import os
import torch

from ModelUpdatingEnv import ModelUpdatingEnv

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)

csv_path = r"PATH"
initial_dataset = pd.read_csv(csv_path)

print(initial_dataset.head())

# -----------------------------
# Build Y (Target) and X (age + decile + sexM)
# -----------------------------
required_cols = ["target", "age", "decile", "sexM"]
missing = [c for c in required_cols if c not in initial_dataset.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Ensure age is numeric
initial_dataset["age"] = pd.to_numeric(initial_dataset["age"], errors="coerce")

# Ensure decile is treated as categorical (string category is safest for get_dummies)
initial_dataset["decile"] = initial_dataset["decile"].astype("category")

# Ensure sexM is numeric
initial_dataset["sexM"] = pd.to_numeric(initial_dataset["sexM"], errors="coerce")

# Drop rows with missing values in the required fields
initial_dataset = initial_dataset.dropna(subset=required_cols).copy()

# Sample 1000 rows for faster training
np.random.seed(144)
initial_dataset = initial_dataset.sample(n=min(1000, len(initial_dataset)), random_state=144).reset_index(drop=True)
print(f"Using {len(initial_dataset)} samples for training")

# Target vector
Y = initial_dataset["target"]

# If Target isn't already 0/1, try to convert it safely
# (e.g., "yes"/"no", "True"/"False", etc.)
if Y.dtype == "object" or str(Y.dtype).startswith("category"):
    Y = Y.astype(str).str.strip().str.lower()
    # common mappings
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
    if set(Y.unique()).issubset(set(mapping.keys())):
        Y = Y.map(mapping)
    else:
        # fallback: factorize to 0/1 if exactly two classes
        codes, uniques = pd.factorize(Y)
        if len(uniques) != 2:
            raise ValueError(f"Target must be binary for LogisticRegression; found classes: {list(uniques)}")
        Y = pd.Series(codes, index=initial_dataset.index)

# One-hot encode decile to split it into separate categories
# Coefficients for decile_* are relative to the dropped baseline category.
X = pd.DataFrame({
    "age": initial_dataset["age"].astype(float),
    "sexM": pd.to_numeric(initial_dataset["sexM"], errors="coerce").astype(int)
})

decile_dummies = pd.get_dummies(initial_dataset["decile"], prefix="decile", drop_first=True)
X = pd.concat([X, decile_dummies], axis=1)

# Keep feature names so you can read coefficients by variable/category
feature_names = X.columns.tolist()

# Validate that Y has both classes
unique_classes = np.unique(Y)
if len(unique_classes) != 2:
    raise ValueError(f"Target must be binary. Found {len(unique_classes)} class(es): {list(unique_classes)}")
print(f"Target classes: {dict(zip(*np.unique(Y, return_counts=True)))}")
print(f"Class balance: {Y.value_counts().to_dict()}")

# Convert Y to numpy array (ensures consistency)
Y = Y.values
X = X.values  # Convert X to numpy array as well

# -----------------------------
# Initial logistic regression
# -----------------------------
logit_model = LogisticRegression(max_iter=2000)
try:
    logit_model.fit(X, Y)
except Exception as e:
    raise RuntimeError(f"Failed to fit initial logistic regression model: {e}")

initial_real_coefficients = logit_model.coef_[0]
probs = logit_model.predict_proba(X)[:, 1]

# Predictions on the initial dataset
predictions = logit_model.predict(X)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y, predictions)
print(f"Initial model accuracy: {accuracy:.4f}")

# Print coefficients with names (age + each decile dummy)
coef_series = pd.Series(initial_real_coefficients, index=feature_names).sort_values(key=np.abs, ascending=False)
print("Initial coefficients (named):")
print(coef_series)

# -----------------------------
# RL training loop
# -----------------------------
# Create environment
print("\n" + "="*60)
print("Starting RL Training")
print("="*60)

update_penalties = [0.01, 0.02, 0.03, 0.04, 0.05]
n_features = X.shape[1]
n_timesteps = 100
n_episodes = 300

print(f"Configuration:")
print(f"  - n_features: {n_features}")
print(f"  - n_timesteps per episode: {n_timesteps}")
print(f"  - n_episodes: {n_episodes}")
print(f"  - Total timesteps: {n_timesteps * n_episodes}")
print(f"  - Update penalties: {update_penalties}\n")

reward_dict = {}

if not os.path.exists('Models'):
    os.makedirs('Models')

for penalty in update_penalties:
    print(f"\nTraining with penalty={penalty}...")
    try:
        env = ModelUpdatingEnv(X, Y, initial_real_coefficients, probs, penalty, n_features, n_timesteps)
        rl_model = PPO("MlpPolicy", env, verbose=1, gamma=0.8, n_steps=1000, batch_size=50)
        rl_model.learn(total_timesteps=n_timesteps * n_episodes)
        rl_model.save(f'Models/penalty_{penalty}.zip')

        rewards = np.array(env.rewards)
        reward_dict[penalty] = rewards
        print(f"✓ Training completed successfully for penalty={penalty}")
    except Exception as e:
        print(f"✗ Training failed for penalty={penalty}: {e}")
        import traceback
        traceback.print_exc()
        continue

if not reward_dict:
    print("\n✗ No training completed successfully. Exiting.")
    exit(1)

print(f"\n✓ Successfully trained {len(reward_dict)} penalty configuration(s)")

np.save('Models/reward_dict.npy', reward_dict)

loaded_reward_dict = np.load('Models/reward_dict.npy', allow_pickle=True).item()

output_dir = 'Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\nGenerating plots...")
for penalty, rewards in loaded_reward_dict.items():
    try:
        total_rewards = [np.sum(rewards[i:i + 100]) for i in range(0, len(rewards), 100)]

        plt.figure()
        plt.plot(total_rewards)
        plt.plot(np.convolve(total_rewards, np.ones((10,)) / 10, mode='valid'), color='r')
        plt.xlabel('Episode', fontsize=18, fontweight="bold")
        plt.ylabel('Reward', fontsize=18, fontweight="bold")
        plt.legend(['Total Reward', 'Rolling Average (10 episodes)'], fontsize=12)
        plt.axhline(y=0, color='g', linestyle='-')

        output_path = os.path.join(output_dir, f'total_reward_per_episode_penalty_{penalty}_100_timesteps.png')
        plt.savefig(output_path)
        plt.close()
        print(f"✓ Saved plot: {output_path}")
    except Exception as e:
        print(f"✗ Failed to generate plot for penalty={penalty}: {e}")

print("\n" + "="*60)
print("Training and evaluation complete!")
print("="*60)