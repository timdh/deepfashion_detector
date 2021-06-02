"""
this file holds all the functions/classes related with model definition
and initialization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init(m):
    """
    initialize the weights of the network
    source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def get_network(net_type, args):
    """
    return the selected network and initialize it,
    raise error if network is not recognized
    """
    if net_type == 'generator':
        net = Generator(l_dim=args.latent_dim, ngf=args.ngf, emb_size=args.emb_size,
                        out_c=args.channels, n_classes=args.n_classes)
    elif net_type == 'discriminator':
        net = Discriminator(ndf=args.ndf, emb_size=args.emb_size, in_c=args.channels,
                            n_classes=args.n_classes, img_size=args.img_size)
    else:
        raise NotImplementedError('network type [%s] is not recognized' % net_type)
    return net.apply(weights_init)

class Generator(nn.Module):
    """
    generator of the GAN
    - DCGAN inspired architecure, extended with conditional input
    """
    def __init__(self, l_dim=100, ngf=64, emb_size=50, out_c=1, n_classes=10):
        super(Generator, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, emb_size),
            nn.Linear(emb_size, 64)
        )
        self.mapping = nn.Sequential(
            # upsample
            nn.ConvTranspose2d(l_dim, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            # upsample
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        self.main = nn.Sequential(
            # upsample, +1 channel for the conditional input
            nn.ConvTranspose2d(ngf*2+1, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, out_c, kernel_size=4, stride=2, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, input, label):
        input = self.mapping(input)
        label = self.embedding(label).view(input.shape[0], -1, input.shape[2], input.shape[3])
        out = torch.cat((input, label), dim=1)
        return self.main(out)

class Discriminator(nn.Module):
    """
    discriminator of the GAN
    - DCGAN inspired architecure, extended with conditional input
    - used spectral normalization for more stable training
    """
    def __init__(self, ndf=64, emb_size=50, in_c=1, n_classes=10, img_size=28):
        super(Discriminator, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, emb_size),
            nn.Linear(emb_size, img_size * img_size)
        )
        self.main = nn.Sequential(
            # downsample, +1 channel for extra conditional input
            spectral_norm(nn.Conv2d(in_c+1, ndf, kernel_size=4, stride=2, padding=3, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # downsample
            spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # downsample
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # final layer
            spectral_norm(nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        label = self.embedding(label).view(input.shape)
        out = torch.cat((input, label), dim=1)
        return self.main(out)
