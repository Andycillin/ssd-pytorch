
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBox(nn.Module):

    def __init__(self):
        super(MultiBox, self).__init__()

    def forward(self, input):
        return []
