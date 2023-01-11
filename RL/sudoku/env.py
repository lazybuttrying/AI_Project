import gym
import torch
from board import Sudoku

class SudokuEnv(gym.Env):
    def __init__(self):
        super(SudokuEnv, self).__init__()
        self.action_space = gym.spaces.Dict(
                {
                    "x": gym.spaces.Discrete(9),
                    "y": gym.spaces.Discrete(9),
                    "v": gym.spaces.Discrete(9),
                }

            )
        self.observation_space = gym.spaces.Box(0,9, shape=(9,9), dtype=int),
        self.reward_range = (0, 27)
        self.sudoku = Sudoku()
        self.times = 0



    def reset(self):
        self.sudoku.reset()
        self.times = 0
        return torch.Tensor(self.sudoku.quest), {}


    def step(self, actions):
        result = self.sudoku.changeValue(actions["x"], actions["y"], actions["v"])
        observation = self.sudoku.quest
        reward = self.sudoku.calcScore(observation)

        done = self.sudoku.isDone()
        truncated = self.times > 10000
        info = {}

        return observation, reward, done, truncated, info



    def render(self, mode='human'):
        pass

    # def close(self):
    #     pass

    # def seed(self, seed=None):
    #     pass

    # def configure(self, *args, **kwargs):
    #     pass

    # def __del__(self):
    #     pass

    # def __str__(self):
    #     pass

    