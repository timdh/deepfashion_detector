"""
This file defines the user arguments of the training/test and networks
"""

import argparse

def get_args(options):
    """
    return the user arguments and print them
    input: a list of the desired options: ['train', 'network', 'generation']
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/')
    if 'train' in options:
        parser.add_argument('--name', type=str, default='exp_name')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
        parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
        parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
        parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
        parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
        parser.add_argument('--print_interval', type=int, default=100, help='interval between loss printing')
        parser.add_argument('--save_interval', type=int, default=5, help='interval between saved models in epochs')
    if 'network' in options:
        parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
        parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
        parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
        parser.add_argument('--channels', type=int, default=1, help='number of image channels')
        parser.add_argument('--ngf', type=int, default=64, help='number of filters of the generator')
        parser.add_argument('--ndf', type=int, default=64, help='number of filters of the discriminator')
        parser.add_argument('--emb_size', type=int, default=50, help='size of the label embedding layer')
    if 'classify' in options:
        parser.add_argument('--n_sample', type=int, default=1000, help='number of sample to be generated')
        parser.add_argument('--not_save_samples', action='store_false', help='if set, do not save samples images to disk')
        parser.add_argument('--dir_gen', type=str, default='samples/generated_samples', help='dir where generated samples are saved')
        parser.add_argument('--dir_real', type=str, default='samples/real_samples', help='dir where real samples from dataset are saved')
        parser.add_argument('--saved_path', type=str, default='exp_name', help='path where the saved model weights are saved')
        parser.add_argument('--model_name', type=str, default='final_trained_generator.pth', help='name of the saved model')
    args = parser.parse_args()
    for arg in vars(args):
        print('-', arg+':', getattr(args, arg))
    print('----------------------------------')
    return args
