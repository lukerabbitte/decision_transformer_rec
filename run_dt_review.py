import logging

from fig_generators.generate_loss_visualisation import plot_loss
from load_data import load_data
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from mingpt.model_review import GPT, GPTConfig
from mingpt.trainer_review import Trainer, TrainerConfig
import argparse
import matplotlib.pyplot as plt
from fig_generators import generate_loss_visualisation

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)  # needed for mingpt
parser.add_argument('--context_length', type=int, default=30)  # the number of tokens in context
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()


class ReviewDataset(Dataset):

    def __init__(self, states, block_size, actions, terminal_indices, returns_to_go, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1  # Vocab size is the space of items to recommend. Used to be 0-indexed, now our actions are 1-indexed.
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
        # print(f"at init point states is a tensor: {torch.is_tensor(states)}")
        # print(f"states size was: {states.size()[0]}")
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        # print(f"actions size was: {actions.size()[0]}")
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        # print(f"returns_to_go size was: {returns_to_go.size()[0]}")
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        # print(f"timesteps for this idx: {idx} are {timesteps.squeeze(0).squeeze(0)}")

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
        self.start_indices = self.terminal_indices
        self.start_indices = np.insert(self.start_indices, 0, 0)

    def __len__(self):
        return len(self.states)

    # Returns user data from a complete matrix of user interactions where they have rated every item, for eval purposes
    # Also note that each of our terminal indices will be one index higher than the last entry for a user.
    # This is in keeping with Python's upper bound exclusive behaviour.
    def __getitem__(self, user_id):
        # print(f"user_id passed to EvaluationDataset getitem is: {user_id}")
        idx = self.start_indices[user_id - 1]
        done_idx = None if user_id == self.start_indices.size else self.terminal_indices[
            user_id - 1]  # avoid array out of limit bug for last user

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
states, actions, returns, terminal_indices, returns_to_go, train_timesteps = load_data(
    "goodreads/goodreads_train_data_1024_users.tsv")
train_dataset = ReviewDataset(states, args.context_length * 3, actions, terminal_indices, returns_to_go, train_timesteps)
len_train_dataset = len(states)
# print(f"max(timesteps) is {max(timesteps)}")

# Test
states, actions, returns, terminal_indices, returns_to_go, timesteps = load_data(
    "goodreads/goodreads_test_data_1024_users.tsv")
test_dataset = ReviewDataset(states, args.context_length * 3, actions, terminal_indices, returns_to_go, timesteps)
len_test_dataset = len(states)
# print(f"max(timesteps) is {max(timesteps)}")

# Eval
states, actions, returns, terminal_indices, returns_to_go, timesteps = load_data(
    "/home/luke/code/decision_transformer_rec/goodreads/goodreads_eval_data_1024_users.tsv")
eval_dataset = EvaluationDataset(states, actions, returns, returns_to_go, terminal_indices, timesteps)

# print(f"max(train_timesteps) is {max(train_timesteps)}")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(train_timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * args.context_length * 3,
                      num_workers=4, seed=args.seed, model_type=args.model_type,
                      ckpt_path="checkpoints/model_checkpoint.pth",
                      max_timestep=max(train_timesteps))
trainer = Trainer(model, train_dataset, test_dataset, eval_dataset, tconf)

train_losses, test_losses = trainer.train()

plot_loss(train_losses, test_losses, args.context_length, args.batch_size, args.model_type,
          mconf.n_layer, mconf.n_head, mconf.n_embd, len_train_dataset, len_test_dataset)

print(f"train_losses: {train_losses}")
print(f"test_losses: {test_losses}")
