import logging

from load_data import load_data
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from mingpt.model_review import GPT, GPTConfig
from mingpt.trainer_review import Trainer, TrainerConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123) # needed for mingpt
parser.add_argument('--context_length', type=int, default=30)  # the number of tokens in context
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()


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

    def __init__(self, states, actions, returns, returns_to_go, terminal_indices, timesteps):
        self.states = states
        self.actions = actions
        self.returns = returns
        self.returns_to_go = returns_to_go
        self.terminal_indices = terminal_indices
        self.timesteps = timesteps

        # Get indices denoting start of each user's interaction trajectory
        user_indices = [0]
        post_terminal_indices = [index + 1 for index in terminal_indices]
        user_indices += post_terminal_indices # start index of each user trajectory

        self.user_indices = user_indices

    def __len__(self):
        return len(self.states)

    # Returns user data from a complete matrix of user interactions where they have rated every item, for eval purposes
    def __getitem__(self, user_id):

        idx = self.user_indices[user_id - 1] # start of user trajectory
        done_idx = self.terminal_indices[user_id - 1] # end of user trajectory

        # Return tensors of (episode_length, 1)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, returns, returns_to_go

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Train
states, actions, returns, terminal_indices, returns_to_go, timesteps = load_data("/home/luke/code/decision_transformer_rec/goodreads/goodreads_train_data.tsv")
train_dataset = ReviewDataset(states, args.context_length * 3, actions, terminal_indices, returns_to_go, timesteps)

states, actions, returns, terminal_indices, returns_to_go, timesteps = load_data("/home/luke/code/decision_transformer_rec/goodreads/goodreads_eval_data.tsv")
eval_dataset = EvaluationDataset(states, actions, returns, returns_to_go, terminal_indices, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * args.context_length * 3,
                      num_workers=4, seed=args.seed, model_type=args.model_type,
                      ckpt_path="checkpoints/model_checkpoint.pth",
                      max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, eval_dataset, tconf)

trainer.train()
