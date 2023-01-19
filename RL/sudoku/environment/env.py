import gym
import torch
import numpy as np
from environment.board import Sudoku

LEFT_TIMES = 200


class SudokuEnv(gym.Env):

    def __init__(self, env_config):
        self.env = Sudoku(env_config["device"] if env_config else {})
        self.left_times = LEFT_TIMES
        self.action_space = gym.spaces.Dict({
            "x": gym.spaces.Discrete(9),
            "y": gym.spaces.Discrete(9),
            "v": gym.spaces.Discrete(9)
        })
        self.observation_space = gym.spaces.Box(
            # np.array(self.env.reset()),
            shape=(9, 9, 1), high=9, low=1)

    def step(self, action):
        self.left_times -= 1
        changed = self.env.updateBoard(action)
        state = self.env.board

        score = self.env.calcScore()
        reward = (score/27.0 if score else -0.5) if changed else -1
        truncated = self.left_times <= 0
        done = reward == 1
        info = {}
        return state, reward, truncated, done, info

    def reset(self, type=None):
        self.left_times = LEFT_TIMES
        return self.env.reset(type=type)

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
