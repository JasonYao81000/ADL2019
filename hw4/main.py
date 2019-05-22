import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

import image_generator

def parse():
    parser = argparse.ArgumentParser(description="pytorch spectral normalization gan on cartoonset")
    parser.add_argument('--seed', type=int, default=9487,
                        help='random seed for torch and numpy')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', 
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual step number (useful on restarts)')
    parser.add_argument('-z', '--z-dim', default=128, type=int, metavar='N',
                        help='dimension of z')
    parser.add_argument('--disc-iter', default=5, type=int, metavar='N',
                        help='number of updates to discriminator for every update to generator')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        choices=['dcgan', 'resnet'],
                        help='model architecture: ' +
                        ' | '.join(['dcgan', 'resnet']) +
                        ' (default: resnet)')
    parser.add_argument('--loss', default='hinge', type=str,
                        help='loss function')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size per process (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='Initial learning rate.')
    parser.add_argument('--display_freq', '-p', default=100, type=int,
                        metavar='N', help='display frequency (default: 100)')
    parser.add_argument('--image_dir', default='./selected_cartoonset100k', type=str, metavar='PATH',
                        help='path to images folder')
    parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, metavar='PATH',
                        help='path to checkpoints folder')
    parser.add_argument('--resume', default='last.ckpt', type=str, metavar='PATH',
                        help='path to the latest checkpoint (default: last.ckpt)')
    parser.add_argument('--ckpt_last', default='last.ckpt', type=str, metavar='PATH',
                        help='path to the latest checkpoint (default: last.ckpt)')
    parser.add_argument('--ckpt_best', default='best.ckpt', type=str, metavar='PATH',
                        help='path to the best checkpoint (default: best.ckpt)')
    parser.add_argument('--save_freq', type=int, default=1, help='saving last model frequency')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--mode', default='train', choices=['train', 'test_fid', 'test_human'],
                        help='mode: ' + ' | '.join(['train', 'test_fid', 'test_human']) + ' (default: train)')
    parser.add_argument('--test_fid_file', type=str, default='./sample_test/sample_fid_testing_labels.txt', help='test fid file path')
    parser.add_argument('--test_human_file', type=str, default='./sample_test/sample_human_testing_labels.txt', help='test human file path')

    args = parser.parse_args()
    return args

def run(args):
    # Fix the random seeds
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create checkpoints folder if it does not exist
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # Create image generator for cartoon dataset.
    datasets = image_generator.Dataset(args.image_dir, args.seed)
    dataloader = torch.utils.data.DataLoader(
        datasets, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True, pin_memory=True)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        print(batch_idx, data.shape, target.shape)
        import matplotlib.pyplot as plt
        plt.imshow(data[0].cpu().data.numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.savefig('1.png')
        exit()

def main():
    # Parse command line and run
    args = parse()
    run(args)

if __name__ == '__main__':
    main()