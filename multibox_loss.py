import torch.nn as nn


class MultiboxLoss(nn.Module):
    def __init__(self, num_of_classes):
        super(MultiboxLoss, self).__init__()
        self.num_of_classes = num_of_classes


