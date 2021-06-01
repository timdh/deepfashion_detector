"""
this file holds the model train loop
"""

import os
import numpy as np
import torch
from torch.autograd import Variable

import options
import visuals
import dataloader
import model


# set the arguments
args = options.get_args()

img_save_path = os.path.join(args.name, 'images')
os.makedirs(img_save_path, exist_ok=True)

C,H,W = args.channels, args.img_size, args.img_size

# Initialize the generator and discriminator
generator = model.get_network(net_type='generator', channels=args.channels)
discriminator = model.get_network(net_type='discriminator', channels=args.channels)

# Load the data
dataloader = dataloader.load_fashionMNIST(args, isTrain=True)

# load visualization board
writer = visuals.Writer(args.name)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# set to CUDA, if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        Batch_Size = args.batch_size
        N_Class = args.n_classes
        img_size = args.img_size
        # Adversarial ground truths
        valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
        fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

        real_y = torch.zeros(Batch_Size, N_Class)
        real_y = real_y.scatter_(1, labels.view(Batch_Size, 1), 1).view(Batch_Size, N_Class, 1, 1).contiguous()
        real_y = Variable(real_y.expand(-1, -1, img_size, img_size).cuda())

        # Sample noise and labels as generator input
        noise = Variable(torch.randn((Batch_Size, args.latent_dim,1,1)).cuda())
        gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
        gen_y = torch.zeros(Batch_Size, N_Class)
        gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).view(Batch_Size, N_Class,1,1).cuda())
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss for real images
        d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
        # Loss for fake images
        gen_imgs = generator(noise, gen_y)
        gen_y_for_D = gen_y.view(Batch_Size, N_Class, 1, 1).contiguous().expand(-1, -1, img_size, img_size)

        d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y_for_D).squeeze(), fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(gen_imgs,gen_y_for_D).squeeze(), valid)
        g_loss.backward()
        optimizer_G.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, args.latent_dim,1,1))).cuda())
            #fixed labels
            y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
            y_fixed = torch.zeros(N_Class**2, N_Class)
            y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).view(N_Class**2, N_Class,1,1).cuda())

            gen_imgs = generator(noise, y_fixed).view(-1,C,H,W)

            visuals.save_im(gen_imgs, img_save_path, epoch, batches_done, args.n_classes)

        if batches_done % args.print_interval == 0:
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                            d_loss.data.cpu(), g_loss.data.cpu()))
            writer.add_losses(g_loss, d_loss, batches_done)

        if epoch > 0 and epoch % args.save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.name, 'epoch_%d.pth' % epoch))

writer.close_writer()
torch.save(generator.state_dict(), os.path.join(args.name, 'final_trained_generator.pth'))
