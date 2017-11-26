'''Convert pretrained VGG model to SSD.
'''
import torch

from ssd import SSD
import config.config as CONFIG

vgg = torch.load('./vgg16_reducedfc.pth')

ssd = SSD('train', CONFIG.CONFIG_SSD_300, 21)

layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

for layer_idx in layer_indices:
    ssd.base[layer_idx].weight.data = vgg['features.%d.weight' % layer_idx]
    ssd.base[layer_idx].bias.data = vgg['features.%d.bias' % layer_idx]

ssd.base.load_state_dict(vgg)
torch.save(ssd.state_dict(), 'ssd.pth')
