import torch


class BoxMatcher:
    def iou(self, boxes1, boxes2):
        """
        Calculate intersection over union for each pair of boxes
        :param boxes1: bounding boxes in form (x1,y1,x2,x2) [n,4]
        :param boxes2: bounding boxes in form (x1,y1,x2,x2) [m,4]
        :return: Tensor [n,m] overlap of n[i] and m[j]
        """
        n = boxes1.size(0)
        m = boxes2.size(0)

        min_xy = torch.max(
            boxes1[:, :2].unsqueeze(1).expand(n, m, 2),
            boxes2[:, :2].unsqueeze(0).expand(n, m, 2)
        )

        max_xy = torch.min(
            boxes1[:, 2:].unsqueeze(1).expand(n, m, 2),
            boxes2[:, 2:].unsqueeze(0).expand(n, m, 2)
        )

        wh = max_xy - min_xy
        wh[wh < 0] = 0  # no overlap
        intersection = wh[:, :, 0] * wh[:, :, 1]  # NxM
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # N
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # M
        # Transform to N x M
        area1.unsqueeze(1).expand_as(intersection)
        area2.unsqueeze(0).expand_as(intersection)

        return intersection / (area1 + area2 - intersection)

    def encode(self, default_boxes, target_boxes, classes, threshold=0.5):
        """
        :param default_boxes: tensor of default boxes (x,y,w,h)
        :param target_boxes: tensor of target boxes (x1,y1,x2,y2)
        :param classes: class labels
        :param threshold: min overlap to be kept
        :return:
        """
        iou = self.iou(target_boxes, self.convert_to_two_point_form(default_boxes))
        max_overlap, max_overlap_idx = iou.max(0)  # Best label for each default box [1, num_of_default_boxes]
        max_overlap.squeeze_(0)
        max_overlap_idx.squeeze_(0)

        best_target_boxes = target_boxes[max_overlap_idx]
        # FIXME: add dynamic variances
        variances = [0.1, 0.2]
        delta_cxcy = (best_target_boxes[:, :2] + best_target_boxes[:, 2:]) / 2 - default_boxes[:, :2]
        delta_cxcy /= variances[0] * default_boxes[:, 2:]
        delta_wh = (best_target_boxes[:, 2:] - best_target_boxes[:, :2]) / default_boxes[:, 2:]
        delta_wh = torch.log(delta_wh) / variances[1]
        target_loc = torch.cat([delta_cxcy, delta_wh], 1)

        target_conf = 1 + classes[max_overlap_idx]  # prepend background class
        target_conf[max_overlap < threshold] = 0  # keep iou > 50% only
        return target_conf, target_loc

    def convert_to_two_point_form(self, boxes):
        """
        Convert boxes in form (cx,cy,w,h) to (x1,y1,x2,y2)
        :param boxes: tensor of n boxes in form (cx,cy,w,h)
        :return: tensor of n boxes in form (x1,y1,x2,y2)
        """
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)

    def nms(self, bboxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.
        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.
        Returns:
          keep: (tensor) selected indices.
        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def decode(self, default_boxes, loc, conf):
        """
        loc, conf back to x1,y1,x2,y2 and labels
        :param loc:
        :param conf:
        :return:
        """
        # FIXME: dynamic variances
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * default_boxes[:, 2:]
        cxcy = loc[:, :2] * variances[0] * default_boxes[:, 2:] + default_boxes[:, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)

        max_conf, labels = conf.max(1)
        ids = labels.squeeze(1).nonzero().squeeze(1)
        keep = self.nms(boxes[ids], max_conf[ids].squeeze(1))

        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]
