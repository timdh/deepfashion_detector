"""
This file handles the visualization part of the training process, ie, plotting 
losses and saving images
"""

import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


def save_im(gen_imgs, save_name, epoch, batches_done, n_class):
    im_name = save_name + '/%d-%d.png' % (epoch, batches_done)
    save_image(gen_imgs.data, im_name, nrow=n_class, normalize=True)

class Writer():
    """
    use tensorboard to plot generator and discriminator losses
    """
    def __init__(self, exp_name):
        self.writer = SummaryWriter(os.path.join(exp_name, 'runs'))

    def add_losses(self, g_loss, d_loss, batches_done):
        self.writer.add_scalar("Loss/train/generator", g_loss, batches_done)
        self.writer.add_scalar('Loss/train/discriminator', d_loss, batches_done)

    def close_writer(self):
        self.writer.flush()
        self.writer.close()
