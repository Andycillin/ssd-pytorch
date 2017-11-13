CONFIG_SSD_300 = {
    'base': {
        'input_channels': 3,  # RGB
        'config': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    },
    'extras': {
        'input_channels': 1024,
        'config': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    },
    'prior_boxes': [4, 6, 6, 6, 4, 4],
    'base_source_maps': [13, -1]
}
