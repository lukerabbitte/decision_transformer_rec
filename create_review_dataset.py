import numpy as np
import pandas as pd
from utils.utils import get_terminal_indices
import os

"""
Returns MovieLens Data in the same form as was done in create_dataset
-> states, actions, returns, terminal indices, returns-to-go, timesteps
"""


def create_review_dataset(dataset_file):
    dataset = pd.read_csv(dataset_file, sep='\t')

    states = dataset.iloc[:, 0]
    actions = dataset.iloc[:, 1]
    returns = dataset.iloc[:, 2]

    terminal_indices = get_terminal_indices(states)

    # -- create reward-to-go dataset
    start_index = 0
    returns_to_go = np.zeros_like(returns)

    for i in terminal_indices:
        curr_traj_returns = returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):
            returns_to_go_j = curr_traj_returns[j - start_index:i - start_index]
            returns_to_go[j] = sum(returns_to_go_j)
        start_index = i

    print('Max return to go is %d' % max(returns_to_go))

    # create timestep dataset
    timesteps = dataset.iloc[:, 3]
    print('Max timestep is %d' % max(timesteps))

    states = np.array(states)
    actions = np.array(actions)
    returns = np.array(returns)
    terminal_indices = np.array(terminal_indices)
    # returns_to_go is already an np array
    timesteps = np.array(timesteps)

    directory = "./data_dumps/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        np.savetxt(directory + 'states.txt', states)
        np.savetxt(directory + 'actions.txt', actions)
        np.savetxt(directory + 'returns.txt', returns)
        np.savetxt(directory + 'terminal_indices.txt', terminal_indices)
        np.savetxt(directory + 'returns_to_go.txt', returns_to_go)
        np.savetxt(directory + 'timesteps.txt', timesteps)

    return states, actions, returns, terminal_indices, returns_to_go, timesteps
