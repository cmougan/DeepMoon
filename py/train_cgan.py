import numpy as np
import argparse
import json
import math
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

grade_names = ['5+',
               '6A', '6A+',
               '6B', '6B+',
               '6C', '6C+',
               '7A', '7A+',
               '7B', '7B+',
               '7C', '7C+',
               '8A', '8A+',
               '8B', '8B+']

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, default='../data/2016+2017.json', help='filename of input JSON data')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_workers', type=int, default=4, help='number of workers')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--d_features', type=int, default=512, help='number of features in the discriminator\'s net')
parser.add_argument('--g_mult', type=int, default=1, help='multiplier for the number of features in the generator\'s net')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU used for training (0-indexed)')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--output_dir', type=str, default='cgan_images', help='directory for generated images')
parser.add_argument('--crop_w', type=int, default=32, help='crop generated images to this width')
parser.add_argument('--crop_h', type=int, default=32, help='crop generated images to this height')
opt = parser.parse_args()
print(opt)

n_classes = len(grade_names)

n_channels = 1 # grayscale image
img_size = 32 # 32x32
img_shape = (n_channels, img_size, img_size)

cuda = torch.cuda.is_available()
os.makedirs(opt.output_dir, exist_ok=True)

g_features_base = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + n_classes, opt.g_mult * g_features_base, normalize=False),
            *block(opt.g_mult * g_features_base, 2 * opt.g_mult * g_features_base),
            *block(2 * opt.g_mult * g_features_base, 4 * opt.g_mult * g_features_base),
            *block(4 * opt.g_mult * g_features_base, 8 * opt.g_mult * g_features_base),
            nn.Linear(8 * opt.g_mult * g_features_base, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), opt.d_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.d_features, opt.d_features),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.d_features, opt.d_features),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.d_features, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    torch.cuda.set_device(opt.gpu_id)
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Create dataset

from dataset import MoonboardProblemDataset

with open(opt.input_json) as input_file:
    json_data = json.load(input_file)

train_set = MoonboardProblemDataset(json_data)

params = {'batch_size': opt.batch_size,
          'shuffle': True,
          'num_workers': opt.n_workers}

dataloader = DataLoader(MoonboardProblemDataset(json_data), **params)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_rows, batches_done):
    """Saves a grid of generated images ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_rows * n_classes, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_rows) for num in range(n_classes)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    gen_imgs = np.array(np.flip(gen_imgs.data[:,:,0:opt.crop_h,0:opt.crop_w].cpu(), axis=2))
    save_image(torch.from_numpy(gen_imgs), os.path.join(opt.output_dir, '%d.png' % batches_done), nrow=n_classes, normalize=True)

# ----------
#  Training
# ----------

print('┌───────────┬───────────┬────────────────────┬────────────────────┐')
print('│   Epoch   │   Batch   │ Discriminator Loss │   Generator Loss   │')
print('├───────────┼───────────┼────────────────────┼────────────────────┤')

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("│ %04d/%04d │ %04d/%04d │ %+.15f │ %+.15f │" % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_rows=20, batches_done=batches_done)

print('└───────────┴───────────┴────────────────────┴────────────────────┘')
