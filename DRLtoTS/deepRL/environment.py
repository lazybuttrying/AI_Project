from typing import Tuple
import pandas as pd
import gym
from gym import spaces
import numpy as np
import torch

from dataset import Dataset
from log import LOGGER


class ForecastYieldEnv(gym.Env):
    """
    # Description


    Source of Dataset : https://aifactory.space/competition/detail/2091


    # Action Space
    Select action which has biggest absolute
    'change' action acts continuous, but others not
        Range : -600 ~ 600 (continuous)
    | Num |Action|
    |-----|------|
    | 0   |change| (up +1, down -1)
    | 1   | hold |
    | 2   | skip |  # NaN is mapped as 0





    # Observation Space
    A observation means a day
        Dimension : 32 x 8 x 3
    | Num | Observation (Layer)   |
    |-----|-----------------------|
    | 0   | First: weather,export |
    | 1   | Second: domae(-somae) |
    | 2   | Third: pummok         |


    # Reward
    The reward is -0.1 every loop and +1000/N for every value forecasted.
        (N == the total number of value forecasted in the episode)
    if you have finished in 732 times in a episode,
        reward is 1000 - 0.1*732 = 926.8 points.

    How much close to true value

    # | Abs | value|
    # |-----|------|
    # | 50  |  50  |
    # | 100 |  25  |
    # | 250 |  10  |
    # | 600 |  1   |
    # | inf | -100 |


    # Episode End
    1. Good Case) Absolute Deviation is under 50
    2. Truncated Case)
        - Absolute Deviation is over 600 -> reward: -100
        - It tried over 1000 times



    # Transition Dynamics


    """

    def __init__(self, net: torch.nn.Module, data: Dataset):
        super().__init__()
        self.net = net
        self.data = data
        self.past_reward = data.dataset[0, 2, -1, 2]
        self.done_perfect = 0

        self.original_obs = np.copy(data.dataset)
        self.observation_space = np.copy(data.dataset)
        # np.copy(self.original_dataset)
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # change, hold, skip

    def reset(self):
        super().reset()
        self.observation_space = np.copy(self.original_obs)
        self.past_reward = self.observation_space[0, 2, -1, 2]
        self.data.data_idx = 0
        self.action_space = spaces.Box(
            np.array([-1, -1, -1]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # change, hold, skip
        return self.data.get_next_line()

    def get_reward(self, actions, vtrue):

        if vtrue == 0 and torch.abs(actions[2]) > 0.5:
            return [-100, -100, 100]
        elif vtrue == self.past_reward and torch.abs(actions[1]) > 0.5:
            return [-100, 100, -100]

        deviation = vtrue - torch.trunc(self.past_reward+actions[0]*1000)

        if deviation <= 50:
            return [50, -100, - 100]
        elif deviation <= 100:
            return [25, -100, - 100]
        elif deviation <= 250:
            return [10, -100, - 100]
        elif deviation <= 600:
            return [1, -100, - 100]
        return [-100, -100, - 100]

    def isDone(self, reward):
        if reward == self.past_reward:
            self.done_perfect += 1
        else:
            self.done_perfect = 0
        return self.done_perfect >= 10  # 10번 이상 연속으로 정답이면 종료

    def step(self, state) -> Tuple:
        vtrue = state[2, 0, -1]
        action_prop = self.net.get_action(state)
        # action = np.random.choice(
        #     np.array([0, 1, 2]), p=action_prop.data.numpy())
        LOGGER.info(f"Action prop: {action_prop.data.numpy()}")
        LOGGER.info(f"{state.shape}, {self.data.data_idx}")

        rewards = self.get_reward(action_prop, vtrue)
        LOGGER.info(f"Reward: {rewards}")
        rewards = self.net.discount_rewards(rewards)
        LOGGER.info(f"Rewards: {rewards}")

        truncated = None
        info = {"vtrue": vtrue,
                "date": state[0, 0, 0],
                "loss": self.net.loss_fn(action_prop, rewards)}

        reward = torch.argmax(rewards)
        done = self.isDone(reward)
        self.past_reward = reward  # 보상이 가장 높은 액션 선택했다고 기록

        observation = self.data.get_next_line()
        return observation, reward, truncated, info, done