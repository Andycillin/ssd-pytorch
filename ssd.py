import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from layers_builder import build_layers_from_cfg
from prior_box import PriorBox
from config.config import *
from layers.multibox_layer import MultiBoxLayer


class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''

    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        """out = scale * x / sqrt(\sum x_i^2)"""
        return self.scale * x / (x.pow(2).sum(dim, keepdim=True).clamp(min=1e-12).rsqrt())


class SSD(nn.Module):
    def __init__(self, phase, config, num_of_classes):
        super(SSD, self).__init__()
        self.img_size = config['dim']
        self.phase = phase
        self.l2norm = L2Norm2d(20)
        self.num_of_classes = num_of_classes
        modules = build_layers_from_cfg(config, num_of_classes)
        self.base = modules['base']
        self.extras = modules['extras']
        self.multibox = modules['multibox']
        self.priors = Variable(PriorBox(CONFIG_PRIOR_BOX_300).forward(), volatile=True)

    def forward(self, input):
        f_maps = []
        x = input
        for i in range(23):  # pass through conv4_3
            x = self.base[i](x)

        m = self.l2norm(x)
        f_maps.append(m)
        for i in range(23, len(self.base)):
            x = self.base[i](x)

        f_maps.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                f_maps.append(x)

        print("passed base and extra layers")
        pred_locs, pred_conf = self.multibox(f_maps)
        return pred_locs, pred_conf
