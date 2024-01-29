import logging

from create_synthetic_review_dataset import create_synthetic_review_dataset
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123) # needed for mingpt
parser.add_argument('--context_length', type=int, default=30)  # the number of tokens in context
parser.add_argument('--epochs', type=int, default=5)  # was 5
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)  # was 500000
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

set_seed(args.seed)
"""
Perhaps the largest change so far is in how we consider actions.

Originally, actions were move left, stay still, move right. 3 options. The vocab size was max(actions)+1.
Now, an action is an item ID.
Does it make sense to keep our vocab size as max(actions)+1, where this means the vocab size is the no. of movies?

State is simply the user ID.
Reward is the rating the user gives.

Other recommender-specific RL papers consider a state that includes some history,
and which implement a far more complicated reward function. Will not do this for now.
"""


class ReviewDataset(Dataset):

    def __init__(self, states, block_size, actions, terminal_indices, returns_to_go, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1  # Vocab size is the space of items to recommend.
        self.states = states
        self.actions = actions
        self.terminal_indices = terminal_indices
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps

    def __len__(self):
        return len(self.states) - self.block_size

    # Gets one context length K (30) worth of data from states, actions, returns_to_go, and timesteps
    def __getitem__(self, idx):
        block_size = self.block_size // 3   # aka, the original context length
        done_idx = idx + block_size
        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = min(i, done_idx)
                break
        idx = done_idx - block_size # get data the size of context length (30) from index up to done index
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, returns_to_go, timesteps

class EvaluationDataset(Dataset):

    def __init__(self, states, block_size, actions, terminal_indices, returns_to_go, timesteps):
        self.block_size = block_size # change this to be episode size, determined by checking dataset.
        self.vocab_size = max(actions) + 1  # 1682 + 1, as vocab size is the space of movies to recommend.
        self.states = states
        self.actions = actions
        self.terminal_indices = terminal_indices
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps

        user_indices = [0]
        post_terminal_indices = [index + 1 for index in terminal_indices]
        user_indices += post_terminal_indices

        self.user_indices = user_indices

    def __len__(self):
        return len(self.states) - self.block_size

    # Gets one episode worth of data from states, actions, returns_to_go, and timesteps
    def __getitem__(self, idx):
        block_size = self.block_size // 3   # aka, the original context length
        done_idx = idx + block_size
        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = min(i, done_idx)
                break
        idx = done_idx - block_size # get data the size of context length (30) from index up to done index
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, returns_to_go, timesteps

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Train
states, actions, returns, terminal_indices, returns_to_go, timesteps = create_synthetic_review_dataset()
train_dataset = ReviewDataset(states, args.context_length * 3, actions, terminal_indices, returns_to_go, timesteps)

# Test
states, actions, returns, terminal_indices, returns_to_go, timesteps = create_synthetic_review_dataset(min_ratings_per_user=273, max_ratings_per_user=273)
test_dataset = ReviewDataset(states, args.context_length * 3, actions, terminal_indices, returns_to_go, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * args.context_length * 3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game,
                      max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, test_dataset, tconf)

trainer.train()
