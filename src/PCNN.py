import torch
from torch import nn
import torch.nn.functional as F

#template for platonic CNN

class PCNN(nn.Module):

    def __init__(self, opt):
        super(PCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=opt['im_chan'], out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 16, 10)  # 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 16)
        x = F.relu(self.fc1(x))
        return x