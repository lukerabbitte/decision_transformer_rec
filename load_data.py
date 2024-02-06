import numpy as np
import os
import pandas as pd


def get_terminal_indices(arr):
    idxs = {}
    for i in reversed(range(len(arr))):
        idxs[arr[i]] = i
    done_idxs = list(idxs.values())
    done_idxs.reverse()
    done_idxs = done_idxs[1:]
    return done_idxs


def load_data(filepath):
    num_items = 273
    groups = 4

    data = pd.read_csv(filepath, delimiter="\t")

    states = np.array(data.iloc[:, 0])
    actions = np.array(data.iloc[:, 1])
    returns = np.array(data.iloc[:, 2])
    timesteps = np.array(data.iloc[:, 3])
    terminal_indices = get_terminal_indices(states)
    start_index = 0
    returns_to_go = np.zeros_like(returns)

    # Generate returns-to-go
    for i in terminal_indices:
        curr_traj_returns = returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):
            returns_to_go_j = curr_traj_returns[j - start_index:i - start_index]
            returns_to_go[j] = sum(returns_to_go_j)
        start_index = i

    terminal_indices = np.array(terminal_indices)
    returns_to_go = np.array(returns_to_go)
    timesteps = np.array(data.iloc[:, 3])

    return states, actions, returns, terminal_indices, returns_to_go, timesteps
