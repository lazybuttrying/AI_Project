from torch import nn, optim
import torch
import torch.nn.functional as F
from log import LOGGER

# http://taewan.kim/post/cnn/


class LayerOutput():
    def conv_width(input_width, filter_width, stride, padding_size):
        return (input_width - filter_width + 2*padding_size)//stride + 1

    def conv_height(input_height, filter_height, stride, padding_size):
        return (input_height - filter_height + 2*padding_size)//stride + 1

    def pool_width(input_row_size, pool_size):
        return input_row_size // pool_size

    def pool_height(input_col_size, pool_size):
        return input_col_size // pool_size


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1)
        # torch.Size([1, 3, 32, 8])
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=1)
        # torch.Size([1, 9, 32, 8])

        # self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        # self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)

        self.fc1 = nn.Linear(2304, 3)
        # torch.Size([2304])
        # flatten 다음  Linear는 (flatten 결과인 모든 것의 곱, output dimension)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        # LOGGER.info(x.size())
        x = torch.tanh(self.conv2(x))

        # x = F.max_pool3d(x, kernel_size=2, stride=2)
        # LOGGER.info(x.size())
        # x = F.leaky_relu(self.conv3(x))
        # x = F.leaky_relu(self.conv4(x))
        # x = F.max_pool3d(x, kernel_size=1, stride=2)
        # LOGGER.info(x.size())
        x = torch.tanh(torch.flatten(x))

        # x = torch.reshape(x, (3, -1))
        # LOGGER.info(x.size())
        x = self.fc1(x)
        # LOGGER.info(x)
        x = torch.tanh(x)
        # [ 184744.9   171871.75 -171191.53]
        # # [ 184744.9   171871.75 -171191.53]
        return x