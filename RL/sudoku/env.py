import gym
import torch
import numpy as np
from board import Sudoku


class SudokuEnv(gym.Env):

    def __init__(self, env_config):
        self.env = Sudoku()
        self.left_times = 500
        self.action_space = gym.spaces.Dict({
            "x": gym.spaces.Discrete(9),
            "y": gym.spaces.Discrete(9),
            "v": gym.spaces.Discrete(9)
        })
        self.observation_space = gym.spaces.Box(
            # np.array(self.env.reset()),
            shape=(9, 9, 1), high=9, low=1)

    def action_space_sample(self):
        return self.action_space.sample()

    def step(self, action):
        self.left_times -= 1
        self.env.updateBoard(action)
        state = self.env.board
        reward = self.env.calcScore()
        truncated = self.left_times <= 0
        done = reward == 27
        info = {}
        return state, reward, truncated, done, info

    def reset(self):
        self.left_times = 500
        return self.env.reset().type(torch.LongTensor)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def configure(self, config):
        return self.env.configure(config)


if __name__ == "__main__":
    env = SudokuEnv({})

    print(env.action_space)
    print(env.observation_space)
    print(env.step({"x": 0, "y": 0, "v": 1}))
    print(env.step({"x": 0, "y": 0, "v": 1}))
