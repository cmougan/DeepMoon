import numpy as np
import argparse
import json

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, default='../data/2016+2017.json', help='filename of input JSON data')
parser.add_argument('--output_img', type=str, default='output.png', help='filename of output image')
parser.add_argument('--n_rows', type=int, default=10, help='number of rows')
parser.add_argument('--n_columns', type=int, default=10, help='number of columns')
parser.add_argument('--crop_w', type=int, default=11, help='crop generated images to this width')
parser.add_argument('--crop_h', type=int, default=18, help='crop generated images to this height')
opt = parser.parse_args()
print(opt)

# Create dataset

from dataset import MoonboardProblemDataset

with open(opt.input_json) as input_file:
    json_data = json.load(input_file)

train_set = MoonboardProblemDataset(json_data)

params = {'batch_size': opt.n_rows * opt.n_columns,
          'shuffle': True,
          'num_workers': 1}

dataloader = DataLoader(MoonboardProblemDataset(json_data), **params)
(imgs, labels) = next(iter(dataloader))

imgs = np.array(np.flip(imgs.data[:,:,0:opt.crop_h,0:opt.crop_w], axis=2))
save_image(torch.Tensor(imgs), opt.output_img, nrow=opt.n_columns, normalize=True)
