"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
from collections import deque
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pad_sequence  # for collator function


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = "checkpoints"
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, eval_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.train_losses = []
        self.test_losses = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def custom_collate(batch):
            # Where our batch is a list of (x, y, r, t) tuples
            x_batch, y_batch, r_batch, t_batch = zip(*batch)
            # print(f"x_batch size before: {len(x_batch)}")
            # if len(x_batch) != 128:
            #     print("state size irregular")
            x_batch = pad_sequence(x_batch, batch_first=True)   # sequences of varying length are padded
            # print(f"x_batch size after: {len(x_batch)}")
            y_batch = pad_sequence(y_batch, batch_first=True)
            r_batch = pad_sequence(r_batch, batch_first=True)
            t_batch = pad_sequence(t_batch, batch_first=True)

            return x_batch, y_batch, r_batch, t_batch

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                collate_fn=custom_collate)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            # for each __getitem__ of dataloader, we know state and want to predict action
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # print(f"the y vector being sent to the model is of size: {y.size()} and looks like:\n{y}")
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    # print(f"\nLoss in {it}:\n {loss}")

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                    mean_loss = float(np.mean(losses))

            if is_train:
                self.train_losses.append(mean_loss)

            if not is_train:
                test_loss = float(np.mean(losses))
                self.test_losses.append(test_loss)
                logger.info("test loss: %f", test_loss)
                return test_loss



        best_loss = float('inf')

        best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # Forget about eval for now. But consider passing through the s, a, rtg
            # self.get_returns(5)  # between 1 - 5x max return in dataset (5 for us)

        # Return losses info for visualisation
        return(self.train_losses, self.test_losses)

    def get_returns(self, ret):

        self.model.train(False)

        # Get 10 unique users
        user_ids = random.sample(range(1, (256 * 4) + 1), 10)

        # Will contain all the total rewards per user episode
        total_rewards = []

        for user_id in user_ids:

            # Get a complete matrix for each user showing their interaction history
            states_user, actions_user, rewards_user, returns_to_go_user = self.eval_dataset[user_id]
            # print(actions_user.size()) # 273
            # print(actions_user)
            rtgs = [ret]
            reward_sum = 0

            actions = []

            for i in range(10):
                # State is simply userID at the moment, so we can start at any arbitrary point
                state = states_user[i]  # shape (b, t)
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                all_states = state if i == 0 else torch.cat([all_states, state], dim=0)

                # Handle initial case where state is just one state and actions are none
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=None if i == 0 else torch.tensor(actions, dtype=torch.long).to(
                                            self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(i, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                                 dtype=torch.int64).to(
                                            self.device)))

                # Find the reward corresponding to the generated action from our eval dataset
                action = sampled_action.cpu().numpy()[0, -1]
                action += 1  # action straight from model is 0-indexed, we want 1-indexed
                print(f"Action for user {user_id} was {action}")
                action_index = np.where(actions_user == action)[0][0]
                reward = rewards_user[action_index] # rewards_user is a simple numpy array so no reshaping needed
                # print(f"Reward for user {user_id} was {reward}")
                reward_sum += reward
                actions += [sampled_action]
                rtgs += [rtgs[-1] - reward]

            total_rewards.append(reward_sum)
            # print(f"Recommended 10 new items to the user of id \"{user_id}\", wanted an accumulative total of {ret}, "
            #       f"got {reward_sum}")

        eval_return = sum(total_rewards) / 10.
        print("Desired average return across ten 10-recommendation sequences: %d, Actual average return: %d" % (50, eval_return))
        self.model.train(True)
        return eval_return