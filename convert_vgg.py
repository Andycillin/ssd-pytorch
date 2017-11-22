'''Convert pretrained VGG model to SSD.
'''
import torch

from ssd import SSD
import config.config as CONFIG

vgg = torch.load('./vgg16_reducedfc.pth')

ssd = SSD('train', CONFIG.CONFIG_SSD_300, 21)
ssd.base.load_state_dict(vgg)
torch.save(ssd.state_dict(), 'ssd.pth')
