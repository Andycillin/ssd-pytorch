from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from ssd import SSD
from utils import progress_bar
from dataset import ListDataset
from multibox_loss import MultiboxLoss
import config.config as CONFIG
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root='./data',
                       list_file='./voc12_train.txt', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

testset = ListDataset(root='./data', list_file='./voc12_test.txt',
                      train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

# Model
net = SSD('train', CONFIG.CONFIG_SSD_300, 21)
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # Convert from pretrained VGG model.
    #net.load_state_dict(torch.load('./model/ssd.pth'))
    print("Start new...")

criterion = MultiboxLoss(21)

if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    net.cuda()
    cudnn.benchmark = True

def test(epoch):
    print('\nTesting...')
    net.eval()
    test_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = Variable(images, volatile=True)
        loc_targets = Variable(loc_targets, require_grad=False)
        conf_targets = Variable(conf_targets, require_grad=False)
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        test_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], test_loss / (batch_idx + 1)))

    # Save checkpoint.
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch + 200):
    test(epoch)
