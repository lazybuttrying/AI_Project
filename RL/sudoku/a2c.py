

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
class A2C(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(A2C, self).__init__()

        self.affine1 = nn.Linear(81, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 3)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    


class Backprop:
    def __init__(self, optimizer, gamma, eps):
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps = eps

        
    def back(self, model):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()

                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)

                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            # reset gradients
            self.optimizer.zero_grad()

            # sum up all the values of policy_losses and value_losses
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            # perform backprop
            loss.backward()
            self.optimizer.step()

            # reset rewards and action buffer
            del model.rewards[:]
            del model.saved_actions[:]