import gym
from gym import spaces
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss


# Generate initial dataset
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)
n_samples = 1000

all_datasets = np.load("Datasets/drifted_datasets.npy", allow_pickle=True).item()

class ModelUpdatingEnv(gym.Env):
    """Custom Environment that applies a different drift each episode."""

    def __init__(self, datasets, update_penalty, n_timesteps):
        super(ModelUpdatingEnv, self).__init__()
        
        # Define action space and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)  # Updated shape
        
        self.datasets = datasets
        self.n_timesteps = n_timesteps
        self.current_step = 0
        
        # Extract initial dataset
        self.X, self.Y = datasets[0]

        # Initialize logistic regression model
        self.model = LogisticRegression()

        # Initialize the dataset and coefficients, as well as logs
        self.accuracy_log = []
        self.episode_accuracy_log = []
        self.episode_precision_log = []
        self.episode_recall_log = []
        self.episode_cel_log = []
        self.update_steps = []
        self.update_steps.append(0)
        self.rewards = []
        self.update_penalty = update_penalty
        self.coefficients_log = []  # To track coefficients over the episode

        # Initialize logs to store metrics for state space
        self.time_since_update = 0
        self.accuracy_diff = 0
        self.precision_diff = 0
        self.recall_diff = 0 
        self.cel_diff = 0
        self.action_log = [0]*1

        # Initialize logs for differences since the last update
        self.accuracy_diff_since_update = 0
        self.precision_diff_since_update = 0
        self.recall_diff_since_update = 0
        self.cell_diff_since_update = 0
        
        self.state = None
        self.current_step = 0

        self.reset()
        
    def step(self, action):
        
        self.X, self.Y = self.datasets[self.current_step]

        # Perform action and calculate reward
        if action == 1:
            # Update model
            self.model.fit(self.X, self.Y)
            self.update_steps.append(self.current_step)
            self.time_since_update = 0
            reward = (accuracy_score(self.Y, self.model.predict(self.X)) - self.episode_accuracy_log[-1] - self.update_penalty)
        else:
            reward = 0
            self.time_since_update += 1

        # Get predictions (always need to predict for the new dataset)
        predictions = self.model.predict(self.X)
        predicted_probs = self.model.predict_proba(self.X)[:, 1]
        
        # Log the current coefficients
        self.coefficients_log.append(self.model.coef_.flatten())

        # Calculate current accuracy
        current_cel = log_loss(self.Y, predicted_probs)
        current_accuracy = accuracy_score(self.Y, predictions)
        precision = precision_score(self.Y, predictions)
        recall = recall_score(self.Y, predictions)

        self.last_cel = self.episode_cel_log[self.update_steps[-1]]
        self.last_accuracy = self.episode_accuracy_log[self.update_steps[-1]]
        self.last_precision = self.episode_precision_log[self.update_steps[-1]]
        self.last_recall = self.episode_recall_log[self.update_steps[-1]]
        self.last_action_coefficients = self.coefficients_log[self.update_steps[-1]]
        
        self.episode_cel_log.append(current_cel)
        self.accuracy_log.append(current_accuracy)
        self.episode_accuracy_log.append(current_accuracy)
        self.episode_precision_log.append(precision)
        self.episode_recall_log.append(recall)

        # Calculate differences between current and previous step
        if len(self.episode_accuracy_log) > 1:
            self.cel_diff = self.episode_cel_log[-2] - current_cel
            self.accuracy_diff = self.episode_accuracy_log[-2] - current_accuracy
            self.precision_diff = self.episode_precision_log[-2] - precision
            self.recall_diff = self.episode_recall_log[-2] - recall
        else:
            self.cel_diff = self.accuracy_diff = self.precision_diff = self.recall_diff = 0

        # Calculate differences since the last update
        self.cel_diff_since_update = self.last_cel - current_cel
        self.accuracy_diff_since_update = self.last_accuracy - current_accuracy
        self.precision_diff_since_update = self.last_precision - precision
        self.recall_diff_since_update = self.last_recall - recall

        # Update the action log
        self.action_log.append(action)
        self.action_log = self.action_log[-1:]

        # State
        self.state = np.concatenate((
            [self.cel_diff, self.accuracy_diff, self.precision_diff, self.recall_diff,
            self.cel_diff_since_update, self.accuracy_diff_since_update,
            self.precision_diff_since_update, self.recall_diff_since_update,
            self.time_since_update],
            [self.last_cel, self.last_accuracy, self.last_precision, self.last_recall]
        ))

        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        self.rewards.append(reward)
        return self.state, reward, done, {}

    def reset(self):
        # Reset to the initial state for a new episode
        self.current_step = 0

        # Reset dataset and coefficients to initial values
        self.X, self.Y = self.datasets[0]
        self.model.fit(self.X, self.Y)


        predictions = self.model.predict(self.X)
        predicted_probs = self.model.predict_proba(self.X)[:, 1]

        self.real_coefficients = self.model.coef_.flatten()

        # Reset accuracy and accuracy diff
        initial_cel = log_loss(self.Y, predicted_probs)
        initial_accuracy = accuracy_score(self.Y, predictions)
        precision = precision_score(self.Y, predictions)
        recall = recall_score(self.Y, predictions)

        self.time_since_update = 0

        self.episode_accuracy_log = []
        self.episode_precision_log = []
        self.episode_recall_log = []
        self.episode_cel_log = []
        self.coefficients_log = []
        self.episode_cel_log.append(initial_cel)
        self.episode_accuracy_log.append(initial_accuracy)
        self.episode_precision_log.append(precision)
        self.episode_recall_log.append(recall)
        self.coefficients_log.append(self.real_coefficients.copy())

        self.last_cel = initial_cel
        self.last_accuracy = initial_accuracy
        self.last_precision = precision
        self.last_recall = recall
        self.last_action_coefficients =  self.model.coef_.flatten()

        # Reset differences
        self.accuracy_diff = 0
        self.precision_diff = 0
        self.recall_diff = 0
        self.cel_diff = 0

        # Reset differences since the last update
        self.accuracy_diff_since_update = 0
        self.precision_diff_since_update = 0
        self.recall_diff_since_update = 0
        self.cel_diff_since_update = 0

        self.action_log = [0] * 1
        
        # Set the initial state
        self.state = np.concatenate((
            [self.cel_diff,
            self.accuracy_diff,
            self.precision_diff,
            self.recall_diff,
            self.cel_diff_since_update,
            self.accuracy_diff_since_update,
            self.precision_diff_since_update,
            self.recall_diff_since_update,
            self.time_since_update],
            [self.last_cel, self.last_accuracy, self.last_precision, self.last_recall]
        ))
        
        return self.state  # Ensure the initial state is returned
    
    def get_accuracy_log(self):
        return self.accuracy_log
    def get_update_steps(self):
        return self.update_steps

