import numpy as np
import gym
from torch import nn, optim
import torch.nn.functional as F
from modelling import CustomModel
import torch

from collections import namedtuple

from dataset import INPUT_SIZE


PG_Args = namedtuple("PG_Args", [
    'len_episode',
    'learning_rate',
    'gamma'  # it expand to length of episode in loss calculation
])

Layer_Size = namedtuple("PG_Args", [
    'input',
    'hidden1',
    'output'
])

default_layer = Layer_Size(
    input=4,
    hidden1=150,
    output=3  # left of Right
)


class PolicyGradient():
    def __init__(self, pg_args: PG_Args,
                 layer_size: Layer_Size = default_layer):
        self.pg = pg_args
        self.model = CustomModel()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.pg.learning_rate)

    def discount_rewards(self, rewards):
        # ex:
        # reward -> reward with discount
        # [50, 49, 48...] -> [50.000, 48.510, 47.044...]
        discount_return = rewards * \
            torch.pow(self.pg.gamma,
                      torch.arange(len(rewards)).float())

        discount_return /= discount_return.max()
        # for stability, normalize between 0 and 1

        return discount_return

    def loss_fn(self, act_prop_pred, reward):
        return -1 * torch.sum(reward * torch.log(act_prop_pred))

    def get_action(self, state):  # on continuous action
        state = state[np.newaxis, :]
        print(state.shape)
        print(torch.from_numpy(state).shape)
        pred_action_prop = self.model(torch.from_numpy(state).float())
        action = np.random.choice(
            np.array([0, 1]), p=pred_action_prop.data.numpy())
        return action, pred_action_prop
