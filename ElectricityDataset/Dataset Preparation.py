import openml
import os
import pickle
import numpy as np
np.random.seed(144)
import random
random.seed(144)
import torch
torch.manual_seed(144)
# Load electricity dataset from OpenML
OPENML_DATASET_ID = 151
dataset = openml.datasets.get_dataset(OPENML_DATASET_ID)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe"
)

df = X.copy()
df['class'] = y

# Convert class column to binary (1 for 'UP', 0 for 'DOWN')
df['class'] = df['class'].astype(str).str.upper().str.strip()
df['class'] = df['class'].apply(lambda x: 1 if x == 'UP' else 0)

# Define rolling window parameters
batch_size = 48 * 21  # 3 weeks of data
step_size = 48 * 7    # Move forward by 1 week (336 observations)
rolling_batches = []

# Generate rolling window batches
for i in range(0, len(df) - batch_size + 1, step_size):  
    batch = df.iloc[i : i + batch_size]  # Get a full 3-week window
    rolling_batches.append(batch)

# Save datasets
if not os.path.exists("Datasets"):
    os.makedirs("Datasets")

# Save first batch separately
with open("Datasets/initial_dataset.pkl", "wb") as f:
    pickle.dump(rolling_batches[0], f)

# Save all rolling window batches
with open("Datasets/drifted_datasets.pkl", "wb") as f:
    pickle.dump(rolling_batches, f)
