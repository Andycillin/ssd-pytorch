import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import config.config as CONFIG
from torch.autograd import Variable
from prior_box import PriorBox

from ssd import SSD
from box_matcher import BoxMatcher
from PIL import Image, ImageDraw

# Load model
net = SSD('test', CONFIG.CONFIG_SSD_300, 21)
net.load_state_dict(torch.load('./ssd.pth'))
net.eval()

# Load test image
img = Image.open('./data/2007_000129.jpg')
img1 = img.resize((300, 300))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
img1 = transform(img1)

# Forward
loc, conf = net(Variable(img1[None, :, :, :], volatile=True))

# Decode
data_encoder = BoxMatcher()
priors = PriorBox(CONFIG.CONFIG_PRIOR_BOX_300).forward()
boxes, labels, scores = data_encoder.decode(priors, loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

draw = ImageDraw.Draw(img)
for box in boxes:
    print(box)
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
img.show()
