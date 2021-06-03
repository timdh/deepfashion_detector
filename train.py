"""
this file holds the model training loop
"""

import os
import numpy as np
import torch
from torch.autograd import Variable

from code import options
from code import visuals
from code import dataloader
from code import model

# set the device, training/model arguments, init the experiment dir and init visual tools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = options.get_args(['train', 'network'])
args.name = os.path.join('checkpoints', args.name)
img_save_path = os.path.join(args.name, 'images')
os.makedirs(img_save_path, exist_ok=True)
writer = visuals.Writer(args.name)

# Initialize the networks, optimizers, loss function and data
generator = model.get_network(net_type='generator', args=args).to(device)
discriminator = model.get_network(net_type='discriminator', args=args).to(device)
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
dataloader = dataloader.load_fashionMNIST(args, isTrain=True)

for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(torch.ones(args.batch_size).to(device), requires_grad=False)
        fake = Variable(torch.zeros(args.batch_size).to(device), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)
        labels = Variable(labels.type(torch.LongTensor)).to(device)

        # Sample noise and labels as generator input
        noise = Variable(torch.randn((args.batch_size, args.latent_dim, 1, 1)).to(device))
        gen_labels = Variable(torch.LongTensor(np.random.randint(0, args.n_classes, args.batch_size))).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss for real images
        d_real_loss = adversarial_loss(discriminator(real_imgs, labels).squeeze(), valid)
        # Loss for fake images
        gen_imgs = generator(noise, gen_labels)
        d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_labels).squeeze(), fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # generator loss
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels).squeeze(), valid)
        g_loss.backward()
        optimizer_G.step()

        # visualization, user output and model saving
        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (args.n_classes**2, args.latent_dim,1,1))).to(device))
            labels = np.array([num for _ in range(args.n_classes) for num in range(args.n_classes)])
            y_labels = Variable(torch.LongTensor(labels)).to(device)
            gen_imgs = generator(noise, y_labels).view(-1, args.channels, args.img_size, args.img_size)

            visuals.save_im(gen_imgs, img_save_path, epoch, batches_done, args.n_classes)

        if batches_done % args.print_interval == 0:
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                            d_loss.data.cpu(), g_loss.data.cpu()))
            writer.add_losses(g_loss.item(), d_real_loss.item(), d_fake_loss.item(), batches_done)

        if epoch > 0 and epoch % args.save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.name, 'gen_epoch_%d.pth' % epoch))
            torch.save(discriminator.state_dict(), os.path.join(args.name, 'dis_epoch_%d.pth' % epoch))

writer.close_writer()
torch.save(generator.state_dict(), os.path.join(args.name, 'final_trained_generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(args.name, 'final_trained_discriminator.pth'))
