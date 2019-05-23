import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import sys
import glob
import os

assert len(sys.argv)==2, "Input the directory of the generated images"

fns = glob.glob(os.path.join(sys.argv[1], "*.png"))
N = 5  # output N images, each contains all 144 combinations
assert len(fns) == 144*N

loader = transforms.Compose([transforms.ToTensor()])

results = torch.zeros(144*N, 3, 128, 128)
for i, fn in enumerate(fns):
    img = loader(Image.open(fn)) # [3, 128, 128]
    f_id = int(os.path.basename(fn).split('.png')[0])
    results[f_id] = img

for i in range(N):
    save_image(results[i*144: (i+1)*144], 'results{}.png'.format(i), nrow=12)

