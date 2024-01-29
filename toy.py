import numpy as np
import pandas as pd
from utils.utils import get_terminal_indices
import datetime

# Now you can use last_indices

dataset = pd.read_csv('ml-100k/u1.base', sep='\t')

states = dataset.iloc[:, 0]
actions = dataset.iloc[:, 1]
returns = dataset.iloc[:, 2]

terminal_indices = get_terminal_indices(states)

# -- create reward-to-go dataset
start_index = 0
returns_to_go = np.zeros_like(returns)

for i in terminal_indices:
    curr_traj_returns = returns[start_index:i]
    for j in range(i-1, start_index-1, -1):
        returns_to_go_j = curr_traj_returns[j-start_index:i-start_index]
        returns_to_go[j] = sum(returns_to_go_j)
    start_index = i

# create timestep dataset
timesteps = dataset.iloc[:, 3]