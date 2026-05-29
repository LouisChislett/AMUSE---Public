import gym
from gym import spaces
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)


def safe_auc_score(y_true, y_pred):
    """Safely compute AUC, returning 0.5 if only one class present."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5  # Default to random classifier performance
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5

def safe_log_loss(y_true, y_pred):
    """Safely compute log loss, clipping predictions to avoid numerical issues."""
    try:
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return log_loss(y_true, y_pred_clipped, labels=[0, 1])
    except Exception:
        return 1.0  # Default high loss for error cases

class ModelUpdatingEnv(gym.Env):
    """Custom Environment that applies a different drift each episode."""

    def __init__(self, initial_X, initial_Y, initial_coefficients, initial_probs, update_penalty, n_features, n_timesteps):
        super(ModelUpdatingEnv, self).__init__()
        
        # Define action space and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)  # AUC and log-loss only
        
        # Initialize the dataset and coefficients, as well as logs
        self.episode_auc_log = []
        self.episode_cel_log = []
        self.update_steps = []
        self.rewards = []
        self.update_penalty = update_penalty
        self.coefficients_log = []  # To track coefficients over the episode

        self.initial_X = initial_X
        self.initial_Y = initial_Y
        self.initial_coefficients = initial_coefficients
        self.initial_probs = initial_probs

        self.X = initial_X.copy()
        self.Y = initial_Y.copy()
        self.real_coefficients = initial_coefficients.copy()
        self.Y_probs = initial_probs.copy()

        # Initialize logistic regression model
        self.model = LogisticRegression(max_iter=2000)

        self.n_features = n_features
        self.n_timesteps = n_timesteps

        
        self.upper_threshold = 1  # upper thresholds for coefficients
        self.lower_threshold = -1  # lower threshold for coefficients

        # Initialize logs to store metrics for state space
        self.auc_diff = 0
        self.cel_diff = 0
        self.action_log = [0]*1

        # Initialize logs for differences since the last update
        self.auc_diff_since_update = 0
        self.cel_diff_since_update = 0
        self.time_since_update = 0
        
        self.state = None
        self.current_step = 0

        self.reset()
        
    def step(self, action):
    # Perform action and calculate reward
        if action == 1:
            # Update model
            try:
                self.model.fit(self.X, self.Y)
                self.update_steps.append(self.current_step)
                self.time_since_update = 0
                # Get predictions
                predicted_probs = self.model.predict_proba(self.X)[:, 1]
                self.last_cel = safe_log_loss(self.Y, predicted_probs)
                self.last_auc = safe_auc_score(self.Y, predicted_probs)
                self.last_action_coefficients = self.model.coef_.flatten() # Save the coefficients at the time of the last action == 1
                reward = (self.last_auc - self.episode_auc_log[-1] - self.update_penalty)
                self.update_steps.append(self.current_step)
            except Exception as e:
                # Model fitting failed, give negative reward and use previous state
                reward = -self.update_penalty
                print(f"Warning: Model fitting failed at step {self.current_step}: {e}")
        else:
            reward = 0
            self.time_since_update += 1

        # Apply gentle drift to coefficients
        drift = np.random.normal(loc=0, scale=0.005, size=self.n_features)
        # Apply drift and clip within bounds
        self.real_coefficients = np.clip(self.real_coefficients + drift, self.lower_threshold, self.upper_threshold)
        # Log the current coefficients
        self.coefficients_log.append(self.real_coefficients.copy())

        # Generate new outcomes with the drifted coefficients
        logits = np.dot(self.X, self.real_coefficients)
        self.Y_probs = 1 / (1 + np.exp(-logits))
        self.Y = np.random.binomial(1, self.Y_probs)

        # Get predictions (always need to predict for the new dataset)
        predicted_probs = self.model.predict_proba(self.X)[:, 1]

        # Calculate current metrics
        current_cel = safe_log_loss(self.Y, predicted_probs)
        current_auc = safe_auc_score(self.Y, predicted_probs)
        self.episode_cel_log.append(current_cel)
        self.episode_auc_log.append(current_auc)

        self.cel_diff = self.episode_cel_log[-2] - current_cel
        self.auc_diff = self.episode_auc_log[-2] - current_auc

        # Calculate differences since the last update
        self.cel_diff_since_update = self.last_cel - current_cel
        self.auc_diff_since_update = self.last_auc - current_auc


        self.action_log.append(action)
        self.action_log = self.action_log[-1:]

        # State
        self.state = np.concatenate((
            [self.cel_diff,
            self.auc_diff,
            self.cel_diff_since_update,
            self.auc_diff_since_update,
            self.time_since_update],
            [self.last_cel, self.last_auc]
        ))

        self.current_step += 1
        done = self.current_step >= self.n_timesteps
        self.rewards.append(reward)
        return self.state, reward, done, {}


    def reset(self):
        # Reset to the initial state for a new episode
        self.current_step = 0

        # Reset dataset and coefficients to initial values
        self.X = self.initial_X.copy()
        self.Y = self.initial_Y.copy()
        self.real_coefficients = self.initial_coefficients.copy()
        self.Y_probs = self.initial_probs.copy()

        # Validate that Y has both classes
        if len(np.unique(self.Y)) < 2:
            raise ValueError(f"Initial Y must have both classes (0 and 1). Found only class(es): {np.unique(self.Y)}")

        try:
            self.model.fit(self.X, self.Y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit initial model during reset: {e}")
            
        # Get initial predictions
        predicted_probs = self.model.predict_proba(self.X)[:, 1]

        # Reset metrics
        initial_cel = safe_log_loss(self.Y, predicted_probs)
        initial_auc = safe_auc_score(self.Y, predicted_probs)

        self.time_since_update = 0
        self.update_steps = []
        self.episode_auc_log = []
        self.episode_cel_log = []
        self.coefficients_log = []
        self.episode_cel_log.append(initial_cel)
        self.episode_auc_log.append(initial_auc)
        self.coefficients_log.append(self.real_coefficients.copy())

        self.last_cel = initial_cel
        self.last_auc = initial_auc
        self.last_action_coefficients = self.model.coef_.flatten()

        # Reset differences
        self.auc_diff = 0
        self.cel_diff = 0

        # Reset differences since the last update
        self.auc_diff_since_update = 0
        self.cel_diff_since_update = 0

        self.action_log = [0] * 1
        
        # Set the initial state
        self.state = np.concatenate((
            [self.cel_diff,
            self.auc_diff,
            self.cel_diff_since_update,
            self.auc_diff_since_update,
            self.time_since_update],
            [self.last_cel, self.last_auc]
        ))
        
        return self.state  # Ensure the initial state is returned
    
    def get_update_steps(self):
        return self.update_steps
