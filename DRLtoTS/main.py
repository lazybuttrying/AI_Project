# %%
from dataset import Dataset
from deepRL.environment import ForecastYieldEnv
from deepRL.policy_net import PG_Args, PolicyGradient

import numpy as np
import torch

EPOCHS = 100


pg = PolicyGradient(PG_Args(121, 0.01, 0.99))
env = ForecastYieldEnv(pg, Dataset())
loss_history = []


# %%
for i in range(EPOCHS):
    losses = []
    env.reset()
    for i in range(pg.pg.len_episode):
        observation, reward, _, info, done = env.step()

        losses.append(info["loss"])

        if done:
            break
    loss_history += losses

# %%
