"""
This file holds dataset related functions/classes
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def load_fashionMNIST(args, isTrain):
    """
    this functions returns the FashionMINST dataloader
    """
    os.makedirs(args.dataroot, exist_ok=True)
    data = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(args.dataroot, train=isTrain, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,) * args.channels, (0.5,) * args.channels)
                    ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    return data

def sample_uniform(args):
    """
    returns a uniform batch of the FashionMNIST test set
    """
    data = torchvision.datasets.FashionMNIST(args.dataroot, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,) * args.channels, (0.5,) * args.channels)
                        ]))
    indices = np.random.randint(len(data), size=args.n_sample)
    subset = torch.utils.data.Subset(data, indices)
    return next(iter(torch.utils.data.DataLoader(subset, batch_size=args.n_sample)))
