from typing import Tuple
import pandas as pd
import gym
from gym import logger, spaces
import numpy as np
import torch

from dataset import Dataset


class ForecastYieldEnv(gym.Env):
    """
    # Description


    Source of Dataset : https://aifactory.space/competition/detail/2091


    # Action Space
    Range : -600 ~ 600 (continuous)
    | Num |Action|
    |-----|------|
    | 0   |change| (up +1, down -1)
    | 1   | hold |
    | 2   | skip |  # at NaN (mapped 0 )



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

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # change, hold, skip

    def get_reward(self, actions, vtrue):
        def get_deviation(action, vtrue):
            match action:
                case 0:
                    return vtrue - (self.past_reward+action*1000)//1
                case 1:
                    return vtrue - self.past_reward
                case 2:
                    return (vtrue == False)

        def calc_deviation(deviation):
            match deviation:
                case True:
                    return 50
                case False:
                    return -50
            if deviation <= 50:
                return 50
            elif deviation <= 100:
                return 25
            elif deviation <= 250:
                return 10
            elif deviation <= 600:
                return 1
            else:
                return -100

        rewards = []
        for action in actions:
            deviation = get_deviation(action, vtrue)
            rewards.append(calc_deviation(deviation))
        return rewards

    def step(self) -> Tuple:

        observation = self.data.get_next_line()
        action, action_prop = self.net.get_action(observation)
        rewards = self.get_reward(action_prop, observation[2, -1, 2])
        rewards = self.net.discount_rewards(rewards)
        print(rewards)

        truncated = None
        info = {"date": observation[0, 0, 0],
                "loss": self.net.loss_fn(action_prop, rewards)}

        reward = torch.argmax(rewards)
        done = reward == self.past_reward
        self.past_reward = reward  # 보상이 가장 높은 액션 선택했다고 기록

        return observation, reward, truncated, info, done
