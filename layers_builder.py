import torch.nn as nn


def build_layers_from_cfg(config, num_of_classes):
    return {
        'base': build_base_layers(config['base']),
        'extras': build_extra_layers(config['extras']),
        'heads': build_classify_head(config, num_of_classes)
    }


def build_base_layers(base_config, batch_norm=False):
    layers = []
    input_depth = base_config['in_channels']
    for cfg in base_config['config']:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layer = nn.Conv2d(input_depth, cfg, kernel_size=3, padding=1)
            if batch_norm:
                layers += [layer, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
            else:
                layers += [layer, nn.ReLU(inplace=True)]
            input_depth = cfg

    pool_5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv_6 = nn.Conv2d(input_depth, 1024, kernel_size=3, padding=6, dilation=6)
    conv_7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool_5, conv_6, nn.ReLU(inplace=True), conv_7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


def build_extra_layers(extra_config):
    layers = []
    input_depth = extra_config['in_channels']
    flag = False
    for index, cfg in enumerate(extra_config['config']):
        if input_depth != 'S':
            if cfg == 'S':
                layers += [nn.Conv2d(input_depth, extra_config['config'][index + 1], kernel_size=(1, 3)[flag], stride=2,
                                     padding=1)]
            else:
                layers += [nn.Conv2d(input_depth, cfg, kernel_size=(1, 3)[flag])]
            flag = not flag
        input_depth = cfg
    return nn.ModuleList(layers)


def build_classify_head(config, num_of_classes):
    loc_layers = []
    conf_layers = []
    sources = config['base_source_maps']
    for index, source in enumerate(sources):
        loc_layers += [
            nn.Conv2d(config['base']['config'][source], config['prior_boxes'][index] * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(config['base']['config'][source], config['prior_boxes'][index] * num_of_classes, kernel_size=3,
                      padding=1)]
    for index, input_depth in enumerate(config['extras']['config'][1::2], 2):
        loc_layers += [nn.Conv2d(input_depth, config['prior_boxes'][index] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(input_depth, config['prior_boxes'][index] * num_of_classes, kernel_size=3, padding=1)]

    return {
        'loc_head': nn.ModuleList(loc_layers),
        'conf_head': nn.ModuleList(conf_layers)
    }
