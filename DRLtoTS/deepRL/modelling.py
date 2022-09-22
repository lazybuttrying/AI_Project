from torch import nn, optim
import torch
import torch.nn.functional as F

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
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=1)
        # self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        # self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(120, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.max_pool3d(x, kernel_size=2, stride=2)

        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = F.softmax(torch.flatten(x))

        x = self.fc1(x)
        return x

    def conv3(inputs, filter, stride=1):
        return nn.Conv3d(inputs, filter,
                         kernel_size=3, stride=stride, padding=1,
                         bias=False)
