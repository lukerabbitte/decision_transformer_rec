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

    data = pd.read_csv(filepath, delimiter="\t")

    states = np.array(data.iloc[:, 0])
    # print(f"states from load_data is {states.size}")
    actions = np.array(data.iloc[:, 1])
    rewards = np.array(data.iloc[:, 2])
    timesteps = np.array(data.iloc[:, 3])
    terminal_indices = get_terminal_indices(states)
    start_index = 0
    returns_to_go = np.zeros_like(rewards)
    returns = [0]

    # Generate returns-to-go
    for i in terminal_indices:
        rewards_by_episode = rewards[start_index:i]
        returns[-1] += sum(rewards_by_episode)
        for j in range(i - 1, start_index - 1, -1):
            rewards_by_reverse_growing_episode = rewards_by_episode[j - start_index:i - start_index]
            returns_to_go[j] = sum(rewards_by_reverse_growing_episode)
        start_index = i

    print(f"returns: {returns}")
    terminal_indices = np.array(terminal_indices)
    returns_to_go = np.array(returns_to_go)
    timesteps = np.array(data.iloc[:, 3])

    return states, actions, returns, terminal_indices, returns_to_go, timesteps
