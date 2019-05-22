import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

import image_generator
import resnet
import dcgan

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
    parser.add_argument('-b', '--batch_size', default=128, type=int,
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
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluation frequency')
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
    
    # Build generator and discriminator
    if args.arch == 'resnet':
        discriminator = resnet.Discriminator().cuda()
        generator = resnet.Generator(args.z_dim).cuda()
    elif args.arch == 'dcgan':
        discriminator = dcgan.Discriminator().cuda()
        generator = dcgan.Generator(args.z_dim).cuda()
    else:
        raise ModuleNotFoundError
    
    # Because the spectral normalization module creates parameters
    # that don't require gradients (u and v), we don't want to optimize these using sgd.
    # We only let the optimizer operate on parameters that _do_ require gradients.
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.5, 0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # Use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    # The fixed z for evaluation
    fixed_z = Variable(torch.randn(args.batch_size, args.z_dim).cuda())

    for epoch in range(args.start_epoch, args.epochs):
        # Training an epoch
        start_time = time.time()
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Skip the last batch
            if images.size(0) != args.batch_size: continue

            # Transfer to CUDA
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
            
            # Update the discriminator
            for _ in range(args.disc_iter):
                z = Variable(torch.randn(args.batch_size, args.z_dim).cuda())
                optim_disc.zero_grad()
                optim_gen.zero_grad()
                if args.loss == 'hinge':
                    disc_loss = nn.ReLU()(1.0 - discriminator(images)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
                elif args.loss == 'wasserstein':
                    disc_loss = -discriminator(images).mean() + discriminator(generator(z)).mean()
                else:
                    disc_loss = nn.BCEWithLogitsLoss()(discriminator(images), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                        nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
                disc_loss.backward()
                optim_disc.step()

            # Update the generator
            z = Variable(torch.randn(args.batch_size, args.z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge' or args.loss == 'wasserstein':
                gen_loss = -discriminator(generator(z)).mean()
            else:
                gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
            gen_loss.backward()
            optim_gen.step()

            print("Epoch[%d/%d], Step[%d/%d], disc loss: %.6f, gen loss: %.6f, elapsed time: %.2fs" % \
                (epoch, args.epochs, batch_idx, len(dataloader), disc_loss.item(), gen_loss.item(), time.time() - start_time), end='\r')
        print()
        scheduler_d.step()
        scheduler_g.step()
        
        if epoch % args.save_freq == 0:
            print('Saving the last models...')
            torch.save(discriminator.state_dict(), os.path.join(args.ckpt_dir, 'disc_' + args.ckpt_last))
            torch.save(generator.state_dict(), os.path.join(args.ckpt_dir, 'gen_' + args.ckpt_last))
        
        if epoch % args.eval_freq == 0:
            print('Evaluating...')
            samples = generator(fixed_z).cpu().data.numpy()[:64]
            fig = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(8, 8)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.transpose((1, 2, 0)) * 0.5 + 0.5)

            plt.savefig('{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

def main():
    # Parse command line and run
    args = parse()
    run(args)

if __name__ == '__main__':
    main()