import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from .modelling import CustomModel
import torch

from collections import namedtuple
from log import LOGGER


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

        # random_discount = torch.arange(len(rewards)).float()

        # pows = torch.pow(self.pg.gamma, random_discount)

        discount_return = torch.mul(torch.Tensor(rewards), self.pg.gamma)
        LOGGER.info("Discount Return: {}".format(discount_return))

        # discount_return /= discount_return.max()
        # for stability, normalize between 0 and 1

        return discount_return

    def loss_fn(self, act_prop_pred, rewards):
        return -1 * torch.sum(rewards * torch.log(act_prop_pred))

    def get_action(self, state):  # on continuous action
        state = state[np.newaxis, :]
        # LOGGER.info(state.shape)
        # LOGGER.info(torch.from_numpy(state).shape)
        pred_action_prop = self.model(torch.from_numpy(state).float())

        return pred_action_prop
