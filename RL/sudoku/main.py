from env import SudokuEnv
from a2c import A2C, Backprop
import torch
from torch import optim
import numpy as np


DEVICE = "cuda:0"
EPISODE = 100
LOG_INTERVAL = 10




env = SudokuEnv()
env.reset()
env.render()


model = A2C(n_observations=81, n_actions=27).to(DEVICE)

optimizier = optim.Adam(model.parameters(), lr=0.001)
GAMMA = 0.99
eps = np.finfo(np.float32).eps.item()

backprop = Backprop(optimizier, GAMMA, eps)

if __name__ == "__main__":
    running_reward = 10
    for i_episode in range(EPISODE):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state = state.to(DEVICE)

            with torch.no_grad():
                action = model(state).max(1)
            state, reward, done, truncated, info = env.step(action)

            if truncated:
                done = True

            model.rewards.append(reward)
            ep_reward += reward

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        backprop.back(model)

        # log results
        if i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, 0))
        #     break