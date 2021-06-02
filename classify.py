"""
this script uses the trained GAN discriminator to classify (fake) generated images
from the GAN generator and (real) images from the test set
"""

import os
import numpy as np
import torch
from torch.autograd import Variable

from code import options
from code import model
from code import dataloader
from code import visuals


def sample_generated(args, save_ims, device):
    """
    synthesize samples from the trained GAN generator
    """
    generator = model.get_network(net_type='generator', args=args).to(device)
    generator.load_state_dict(torch.load(os.path.join(args.saved_path, args.model_name)))
    noise = Variable(torch.randn((args.n_sample, args.latent_dim, 1, 1)).to(device))
    gen_labels = Variable(torch.LongTensor(np.random.randint(0, args.n_classes, args.n_sample))).to(device)
    gen_imgs = generator(noise, gen_labels)

    if save_ims:
        os.makedirs(args.dir_gen)
        for i, im in enumerate(gen_imgs):
            visuals.save_im(im, args.dir_gen, str(i).zfill(4), gen_labels[i].item())

    return (gen_imgs, gen_labels)

def sample_dataset_uniform(args, save_ims, device):
    """
    uniformly sample images from the FashionMNIST test set
    """
    real_ims, real_labels = dataloader.sample_uniform(args)

    if save_ims:
        os.makedirs(args.dir_real)
        for i, im in enumerate(real_ims):
            visuals.save_im(im, args.dir_gen, str(i).zfill(4), real_labels[i].item())

    return (real_ims.to(device), real_labels.to(device))

def classify(args, samples, device, tag=None):
    """
    let the discriminator predict whether images in the batch are real or fake
    """
    discriminator = model.get_network(net_type='discriminator', args=args).to(device)
    prediction = discriminator(samples[0], samples[1])
    # print mean and std of batch prediction
    if tag is not None:
        print(tag)
    print('mean of prediction:', prediction.mean().item())
    print('std of prediction:', prediction.std().item())
    print('-------------')
    return torch.round(prediction).sum().item()


args = options.get_args(['network', 'classify'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fake_batch_gt = torch.zeros(args.n_sample, requires_grad=False).to(device)
fake_batch = sample_generated(args, args.not_save_samples, device)

real_batch_gt = torch.ones(args.n_sample, requires_grad=False).to(device)
real_batch = sample_dataset_uniform(args, args.not_save_samples, device)

results_fake = args.n_sample - classify(args, fake_batch, fake_batch_gt, tag='fake')
results_real = classify(args, real_batch, real_batch_gt, tag='real')

accuracy = (results_fake + results_real) / (args.n_sample * 2)
print(str(accuracy) + '% was accurately classified from a balanced (real/fake) set of', str(args.n_sample*2), 'samples')
