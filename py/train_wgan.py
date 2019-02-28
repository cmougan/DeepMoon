import numpy as np
import argparse
import json
import math
import os

from collections import deque

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

torch.set_printoptions(threshold=10000,
                       linewidth=10000,
                       precision=3)

np.set_printoptions(suppress=True,
                    linewidth=np.nan,
                    threshold=np.nan,
                    precision=3)

from dataset import MoonboardProblemDataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, default='../data/2016+2017.json', help='filename of input JSON data')
parser.add_argument('--n_epochs', type=int, default=100000, help='number of epochs of training')
parser.add_argument('--ni_epochs', type=int, default=2000, help='number of non-improving epochs before early stopping')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_workers', type=int, default=4, help='number of workers')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--lambda_start', type=float, default=5, help='lambda for start count loss')
parser.add_argument('--lambda_top', type=float, default=1, help='lambda for top count loss')
parser.add_argument('--d_features', type=int, default=512, help='number of features in the discriminator\'s net')
parser.add_argument('--g_mult', type=int, default=1, help='multiplier for the number of features in the generator\'s net')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU used for training (0-indexed)')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--output_dir', type=str, default='wgan_images', help='directory for generated images')
parser.add_argument('--crop_w', type=int, default=11, help='crop images to this width')
parser.add_argument('--crop_h', type=int, default=18, help='crop images to this height')
parser.add_argument('--disable_visdom', help='disable visdom visualisation', action ='store_true')
parser.add_argument('--visdom_horizon', type=int, default=5000, help='show last n epochs in visdom')
opt = parser.parse_args()
print(opt)
print()

n_channels = 1 # grayscale image
img_size = 32 # 32x32
img_shape = (n_channels, img_size, img_size)
g_features_base = 128

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(opt.gpu_id)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.makedirs(opt.output_dir, exist_ok=True)

# ----------
#  Modules
# ----------

class Normalize(nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super(Normalize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, t):
        t = torch.clamp(t, min=0)
        min_t,_ = torch.min(t, dim=3, keepdim=True)
        max_t,_ = torch.max(t, dim=3, keepdim=True)
        min_t,_ = torch.min(min_t, dim=2, keepdim=True)
        max_t,_ = torch.max(max_t, dim=2, keepdim=True)
        min_t = min_t.expand_as(t)
        max_t = max_t.expand_as(t)
        return self.min_val + torch.mul(torch.div(t - min_t, max_t - min_t), self.max_val - self.min_val)

class Discretize(nn.Module):
    def __init__(self, n_levels=4):
        super(Discretize, self).__init__()
        self.n_levels = n_levels

    def forward(self, t):
        t = torch.mul(t, self.n_levels - 1) # take into account level 0 as well
        t = torch.round(t)
        t = torch.div(t, self.n_levels - 1) # take into account level 0 as well
        return t

class CountPixel(nn.Module):
    def __init__(self, pixel):
        super(CountPixel, self).__init__()
        self.pixel = pixel

        self.model = nn.Sequential(
            Normalize(),
            Discretize()
        )

    def forward(self, t):
        matrix = Tensor(size=t.shape)
        matrix.fill_(self.pixel)
        return torch.sum(torch.eq(self.model(t), matrix), dim=[1,2,3]).type(Tensor)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, opt.g_mult * g_features_base, normalize=False),
            *block(opt.g_mult * g_features_base, 2 * opt.g_mult * g_features_base),
            *block(2 * opt.g_mult * g_features_base, 4 * opt.g_mult * g_features_base),
            *block(4 * opt.g_mult * g_features_base, 8 * opt.g_mult * g_features_base),
            nn.Linear(8 * opt.g_mult * g_features_base, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize count modules
count_start = CountPixel(MoonboardProblemDataset.start)
count_top = CountPixel(MoonboardProblemDataset.top)

# Count loss function
count_loss = nn.L1Loss()
#count_loss = nn.MSELoss()

# ----------
#  Create dataset
# ----------

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

if cuda:
    generator.cuda()
    discriminator.cuda()
    count_start.cuda()
    count_top.cuda()

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if opt.disable_visdom:
    pass
else:
    # ----------
    #  Initialise Visdom
    # ----------
    import visdom
    vis = visdom.Visdom()
    opts=dict(title='Wasserstein GAN with Count Loss (λₛ = ' + str(opt.lambda_start) + ', λₜ = ' + str(opt.lambda_top) + ')',
              xlabel='Epoch',
              width=1900,
              height=600,
              ytickstep=1,
              legend=['Discr. Loss', 'Gener. Loss', 'Average #Start', 'Average #Top'],
              showlegend=True)
    plot = vis.line(Y=np.column_stack((0, 0, 0, 0)), X=np.column_stack((0, 0, 0, 0)), opts=opts)
    d_loss_deque = deque(maxlen=opt.visdom_horizon)
    g_loss_deque = deque(maxlen=opt.visdom_horizon)
    s_deque = deque(maxlen=opt.visdom_horizon)
    t_deque = deque(maxlen=opt.visdom_horizon)
    epoch_deque = deque(maxlen=opt.visdom_horizon)

# ----------
#  Training
# ----------

print('┌───────────────┬───────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬───────────────┐')
print('│     Epoch     │     Batch     │ Discriminator Loss │  Best Discr. Loss  │   Generator Loss   │   Best Gen. Loss   │  N.I. Epochs  │')
print('├───────────────┼───────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼───────────────┤')

batches_done = 0
best_d_loss = math.inf
best_g_loss = math.inf
ni_epochs = 0

top_target_value = 1
start_target_value = 2

for epoch in range(opt.n_epochs):
    best_epoch_d_loss = math.inf
    best_epoch_g_loss = math.inf
    best_epoch_start_count = math.inf
    best_epoch_top_count = math.inf
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)
        fake_imgs_cropped = fake_imgs[:,:,0:opt.crop_h,0:opt.crop_w]
        nd = nn.Sequential(Normalize(), Discretize())
        fake_imgs_cropped_nd = nd(fake_imgs_cropped)

        # Real images
        real_validity = discriminator(real_imgs)

        # Fake images
        fake_validity = discriminator(fake_imgs)

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        # Count loss
        count_start_cropped = count_start(fake_imgs_cropped)
        count_top_cropped = count_top(fake_imgs_cropped)
        start_target = Tensor(imgs.shape[0])
        start_target.fill_(start_target_value)
        top_target = Tensor(imgs.shape[0])
        top_target.fill_(top_target_value)

        count_start_mean = torch.mean(count_start_cropped).item()
        count_top_mean = torch.mean(count_top_cropped).item()

        if abs(count_start_mean - start_target_value) < abs(best_epoch_start_count - start_target_value):
            best_epoch_start_count = count_start_mean

        if abs(count_top_mean - top_target_value) < abs(best_epoch_top_count - top_target_value):
            best_epoch_top_count = count_top_mean

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty \
            + opt.lambda_start * count_loss(count_start_cropped, start_target) + opt.lambda_top * count_loss(count_top_cropped, top_target)

        if abs(d_loss.item()) < best_epoch_d_loss:
            best_epoch_d_loss = abs(d_loss.item())

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            fake_imgs_cropped = fake_imgs[:,:,0:opt.crop_h,0:opt.crop_w]
            count_start_cropped = count_start(fake_imgs_cropped)
            count_top_cropped = count_top(fake_imgs_cropped)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity) \
                - opt.lambda_start * count_loss(count_start_cropped, start_target) - opt.lambda_top * count_loss(count_top_cropped, top_target)

            if abs(g_loss.item()) < best_epoch_g_loss:
                best_epoch_g_loss = abs(g_loss.item())

            g_loss.backward()
            optimizer_G.step()

            print ("│ %06d/%06d │ %06d/%06d │ %+18.13f │ %18.13f │ %+18.13f │ %18.13f │ % 13d │" % (epoch + 1,
                                                                                                      opt.n_epochs,
                                                                                                      i + 1,
                                                                                                      len(dataloader),
                                                                                                      d_loss.item(),
                                                                                                      best_d_loss,
                                                                                                      g_loss.item(),
                                                                                                      best_g_loss,
                                                                                                      ni_epochs))

            if batches_done % opt.sample_interval == 0:
                grid_size = int(math.sqrt(imgs.shape[0]))
                flipped = np.array(np.flip(fake_imgs_cropped.data[:(grid_size * grid_size)].cpu(), axis=2))
                flipped_nd = np.array(np.flip(fake_imgs_cropped_nd.data[:(grid_size * grid_size)].cpu(), axis=2))
                interleaved = np.empty((2 * grid_size * grid_size, 1, opt.crop_h, opt.crop_w), dtype=flipped.dtype)
                interleaved[0::2] = flipped
                interleaved[1::2] = flipped_nd
                save_image(torch.from_numpy(interleaved), os.path.join(opt.output_dir, '%d.png' % batches_done), nrow=2*grid_size, normalize=False)

            batches_done += opt.n_critic

    if opt.disable_visdom:
        pass
    else:
        d_loss_deque.append(best_epoch_d_loss)
        g_loss_deque.append(best_epoch_g_loss)
        s_deque.append(best_epoch_start_count)
        t_deque.append(best_epoch_top_count)
        epoch_deque.append(epoch)
        vis.line(np.column_stack((d_loss_deque, g_loss_deque, s_deque, t_deque)),
            np.column_stack((epoch_deque, epoch_deque, epoch_deque, epoch_deque)), win=plot, update='replace')

    if best_epoch_d_loss < best_d_loss or best_epoch_g_loss < best_g_loss:
        best_d_loss = min(best_d_loss, best_epoch_d_loss)
        best_g_loss = min(best_g_loss, best_epoch_g_loss)
        ni_epochs = 0
    else:
        if ni_epochs < opt.ni_epochs:
            ni_epochs += 1
        else:
            break

print('└───────────────┴───────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴───────────────┘')
