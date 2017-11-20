import torch
import torch.nn as nn


class MultiBoxLayer(nn.Module):
    def __init__(self, config, num_of_classes):
        super(MultiBoxLayer, self).__init__()
        loc_layers = []
        conf_layers = []
        self.num_of_classes = num_of_classes
        sources = config['base_source_maps']
        for index, source in enumerate(sources):
            loc_layers += [
                nn.Conv2d(config['base']['config'][source], config['prior_boxes'][index] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(config['base']['config'][source], config['prior_boxes'][index] * num_of_classes,
                          kernel_size=3,
                          padding=1)]
        for index, input_depth in enumerate(config['extras']['config'][1::2], 2):
            loc_layers += [nn.Conv2d(input_depth, config['prior_boxes'][index] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(input_depth, config['prior_boxes'][index] * num_of_classes, kernel_size=3, padding=1)]

        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)

    def forward(self, f_maps):
        """
        :param x: List of feature maps
        :return:
            loc_preds : predicted offset
            conf_preds : predicted class confidences
        """
        locs = []
        confs = []
        for index, f_map in enumerate(f_maps):
            p_loc = self.loc_layers[i][f_map]
            num = locs.size(0)
            p_loc = p_loc.permute(0, 2, 3, 1).contiguous().view(num, -1, 4)
            locs.append(p_loc)

            p_conf = self.conf_layers[i][f_map]
            p_conf = p_conf.permute(0, 2, 3, 1).contiguous().view(num, -1, self.num_of_classes)
            confs.append(p_conf)

        pred_locs = torch.cat(locs, 1)
        pred_confs = torch.cat(confs, 1)
        return pred_locs, pred_confs
