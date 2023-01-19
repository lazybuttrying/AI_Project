from environment.env import SudokuEnv
import random
from torch import nn
import torch.nn.functional as F
import torch


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super(ActorCritic, self).__init__()
        self.step_size = 1e-5
        # self.discount = 0.99
        self.gamma = 0.9
        # self.epsilon = 0.9

        self.conv1 = nn.Conv1d(9, 81, 1, stride=2, padding=1)
        self.conv2 = nn.Conv1d(40, 729, 1, stride=2, padding=1)
        # self.conv3 = nn.Conv1d(729, 364, 1, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.actor_x_fc1 = nn.Linear(3276, 9)
        self.actor_y_fc1 = nn.Linear(3276, 9)
        self.actor_v_fc1 = nn.Linear(3276, 9)

        self.l3 = nn.Linear(3276, 81)
        self.critic_fc1 = nn.Linear(81, 1)


        self.optimizer = torch.optim.RAdam(
            self.parameters(), lr=self.step_size,)
        # weight_decay=1e-5)
        self.criteria = nn.MSELoss()

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        y = self.maxpool(x)
        y = torch.tanh(torch.flatten(y))

        actor = {
            "x": F.log_softmax(self.actor_x_fc1(y), dim=0),
            "y": F.log_softmax(self.actor_y_fc1(y), dim=0),
            "v": F.log_softmax(self.actor_v_fc1(y), dim=0)
        }
        c = torch.tanh(self.l3(y.detach()))
        critic = torch.tanh(self.critic_fc1(c))

        return actor, critic


if __name__ == "__main__":
    m = ActorCritic()

    target = torch.tensor([3., 5., 7.], requires_grad=True)
    inputs = torch.rand(9, 9, 1)
    actor, critic = m.get_action(inputs)

    creiterion = nn.MSELoss()
    loss = creiterion(torch.argmax(actor[0], keepdim=True).float(
    ), torch.tensor([3.], requires_grad=True))

    print(actor, critic)

    # print(inputs.grad, loss)
