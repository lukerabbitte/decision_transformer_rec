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
    done_idxs.append(len(arr))
    return done_idxs


def load_data(filepath):

    data = pd.read_csv(filepath, delimiter="\t")

    states = np.array(data.iloc[:, 0])
    # print(f"states from load_data is {states.size}")
    actions = np.array(data.iloc[:, 1])
    # Zero-index the actions when loading in
    actions = np.array(data.iloc[:, 1]) - 1
    rewards = np.array(data.iloc[:, 2])
    timesteps = np.array(data.iloc[:, 3])
    terminal_indices = get_terminal_indices(states)
    start_index = 0
    returns_to_go = np.zeros_like(rewards)
    returns = np.array([])

    # Generate returns-to-go
    for i in terminal_indices:
        rewards_by_episode = rewards[start_index:i]
        # print(f"rewards_by_episode: {rewards_by_episode} for data up to done_idx: {i}\n")
        returns = np.append(returns, sum(rewards_by_episode))
        for j in range(i - 1, start_index - 1, -1):
            rewards_by_reverse_growing_episode = rewards_by_episode[j - start_index:i - start_index]
            # print(f"rewards_by_reverse_growing_episode: {rewards_by_reverse_growing_episode}\n")
            returns_to_go[j] = sum(rewards_by_reverse_growing_episode)
            # print(f"returns_to_go: {returns_to_go}")
        start_index = i

    terminal_indices = np.array(terminal_indices)
    returns_to_go = np.array(returns_to_go)
    timesteps = np.array(data.iloc[:, 3])

    # print(f"returns: {returns}")
    # print(f"returns_to_go: {returns_to_go}")

    return states, actions, returns, terminal_indices, returns_to_go, timesteps
