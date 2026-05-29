import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import torch

from ModelUpdatingEnv import ModelUpdatingEnv

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)


#Reinforcement learning
data = np.load("Datasets/initial_dataset.npz")
X = data['X']
Y = data['Y']
probs = np.load("Datasets/probs.npy")
initial_real_coefficients = np.load("Datasets/initial_real_coefficients.npy")


# List of update penalties
update_penalties = [0.01,0.02,0.03,0.04,0.05]
n_features = X.shape[1]
n_timesteps = 100
n_episodes = 300

# Dictionary to store rewards for each penalty
reward_dict = {}

# Create a Models folder if it does not exist
if not os.path.exists('Models'):
    os.makedirs('Models')

# Loop through each update_penalty, train the model, and save it
for penalty in update_penalties:
    env = ModelUpdatingEnv(X, Y, initial_real_coefficients, probs, penalty, n_features, n_timesteps)
    model = PPO("MlpPolicy", env, verbose=1, gamma = 0.8, n_steps=1000, batch_size=50)
    model.learn(total_timesteps=n_timesteps * n_episodes)
    # save the model in the Models folder
    model.save(f'Models/penalty_{penalty}.zip')

    
    # Store the rewards in the dictionary
    rewards = np.array(env.rewards)
    reward_dict[penalty] = rewards

# Save the rewards dictionary to a file
np.save('Models/reward_dict.npy', reward_dict)

# Load the rewards dictionary from the file
loaded_reward_dict = np.load('Models/reward_dict.npy', allow_pickle=True).item()

# Specify the outputs directory relative to the current directory
output_dir = 'Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# plot the reward curve for all timesteps, using groups of 100
for penalty, rewards in loaded_reward_dict.items():
    total_rewards = [np.sum(rewards[i:i + 100]) for i in range(0, len(rewards), 100)]
    
    # Plot the total rewards
    plt.figure()
    plt.plot(total_rewards)
    plt.plot(np.convolve(total_rewards, np.ones((10,)) / 10, mode='valid'), color='r')
    plt.xlabel('Time Step (Hundreds)', fontsize=18, fontweight="bold")
    plt.ylabel('Reward', fontsize=18, fontweight="bold")
    plt.legend(['Total Reward', '100 Timestep Rolling Average'], fontsize=12)
    plt.axhline(y=0, color='g', linestyle='-')

    # Save the plot to the 'outputs' subfolder
    output_path = os.path.join(output_dir, f'total_reward_per_episode_penalty_{penalty}_100_timesteps.png')
    plt.savefig(output_path)

    plt.close()
