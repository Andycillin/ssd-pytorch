import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class MultiboxLoss(nn.Module):
    def __init__(self, num_of_classes):
        super(MultiboxLoss, self).__init__()
        self.num_of_classes = num_of_classes

    def cross_entropy_no_av(self, p, t):
        """ Cross entropy loss for each box w/o averaging
        """
        max = p.data.max()
        return torch.log(torch.sum(torch.exp(p - max), 1)) + max - p.gather(1, t.view(-1, 1))

    def forward(self, loc_p, loc_t, conf_p, conf_t):

        batch_size, num_boxes, _ = loc_p.size()
        positives = conf_t > 0


        num_matched = positives.data.long().sum()
        if num_matched == 0:
            return Variable(torch.Tensor([0]))

        # localization loss
        pos_mask = positives.unsqueeze(2).expand_as(loc_p)
        pos_loc_pred = loc_p[pos_mask].view(-1, 4)
        pos_loc_target = loc_t[pos_mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_pred, pos_loc_target, size_average=False)

        # hard negative mining
        c_loss = self.cross_entropy_no_av(conf_p.view(-1, self.num_of_classes), conf_t.view(-1))

        batch_size, num_boxes = positives.size()
        c_loss[positives] = 0
        c_loss = c_loss.view(batch_size, -1)

        _, idx = c_loss.sort(1, descending=True)
        _, rank = idx.sort(1)

        num_pos = positives.long().sum(1)  # [N,1]
        num_neg = torch.clamp(num_pos * 3, max=num_boxes - num_pos)

        negatives = rank < num_neg.expand_as(rank)

        # confidence loss
        pos_mask = positives.unsqueeze(2).expand_as(conf_p)
        neg_mask = negatives.unsqueeze(2).expand_as(conf_p)
        mask = (pos_mask + neg_mask).gt(0)

        pos_neg = (positives + negatives).gt(0)
        preds = conf_p[mask].view(-1, self.num_of_classes)
        targets = conf_t[pos_neg]
        c_loss = F.cross_entropy(preds, targets, size_average=False)

        loc_loss /= num_matched
        c_loss /= num_matched
        print ("loss calculated")
        print('%f %f' % (loc_loss.data[0], c_loss.data[0]))
        return loc_loss + c_loss
