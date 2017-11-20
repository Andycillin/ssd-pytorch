CONFIG_SSD_300 = {
    'dim': 300,
    'base': {
        'input_channels': 3,  # RGB
        'config': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512],
        'config2': ['M', 512, 512, 512]
    },
    'extras': {
        'input_channels': 1024,
        'config': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    },
    'prior_boxes': [4, 6, 6, 6, 4, 4],
    'base_source_maps': [13, -1]
}

CONFIG_PRIOR_BOX_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v2',
}
