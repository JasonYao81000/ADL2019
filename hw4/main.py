import argparse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import image_generator
import acgan

def parse():
    parser = argparse.ArgumentParser(description="pytorch spectral normalization gan on cartoonset")
    parser.add_argument('--seed', type=int, default=9487,
                        help='random seed for torch and numpy')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', 
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual step number (useful on restarts)')
    parser.add_argument('-z', '--z-dim', default=100, type=int, metavar='N',
                        help='dimension of z')
    parser.add_argument('--disc-iter', default=5, type=int, metavar='N',
                        help='number of updates to discriminator for every update to generator')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='acgan',
                        choices=['acgan', 'wacgan'],
                        help='model architecture: ' +
                        ' | '.join(['acgan', 'wacgan']) +
                        ' (default: acgan)')
    parser.add_argument('--loss', default='bce', type=str,
                        help='loss function')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size per process (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='Initial learning rate.')
    parser.add_argument("--b1", default=0.5, type=float, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", default=0.999, type=float, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--clip_value", default=0.01, type=float, help="lower and upper clip value for disc. weights")
    parser.add_argument('--display_freq', '-p', default=100, type=int,
                        metavar='N', help='display frequency (default: 100)')
    parser.add_argument('--image_dir', default='./selected_cartoonset100k', type=str, metavar='PATH',
                        help='path to images folder')
    parser.add_argument('--eval_dir', default='./eval_images', type=str, metavar='PATH',
                        help='path to evaluation folder')
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

    # Create checkpoints and evaluation folder if it does not exist
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    # Create image generator for cartoon dataset.
    datasets = image_generator.Dataset(args.image_dir, args.seed)
    dataloader = torch.utils.data.DataLoader(
        datasets, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True, pin_memory=True)
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss().cuda()
    auxiliary_loss = torch.nn.CrossEntropyLoss().cuda()

    # Build generator and discriminator, then initialize weights.
    if args.arch == 'acgan':
        generator = acgan.Generator(args.z_dim, datasets).cuda()
        discriminator = acgan.Discriminator(datasets).cuda()
        generator.apply(acgan.weights_init_normal)
        discriminator.apply(acgan.weights_init_normal)
    elif args.arch == 'wacgan':
        # TODO
        raise NotImplementedError
    else:
        raise ModuleNotFoundError

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # The fixed z for evaluation
    fixed_z = Variable(torch.cuda.FloatTensor(
        np.random.normal(0, 1, 
        (len(datasets.attr_hair) * len(datasets.attr_eye) * len(datasets.attr_face) * len(datasets.attr_glasses), 
        args.z_dim))))

    epoch_g_losses = []
    epoch_d_losses = []
    epoch_d_acc_hair = []
    epoch_d_acc_eye = []
    epoch_d_acc_face = []
    epoch_d_acc_glasses = []
    epoch_time = []
    for epoch in range(args.start_epoch, args.epochs):
        # Training an epoch
        start_time = time.time()
        batch_g_losses = []
        batch_d_losses = []
        batch_d_accs_hair = []
        batch_d_accs_eye = []
        batch_d_accs_face = []
        batch_d_accs_glasses = []
        for batch_idx, (images, labels, hair_idxes, eye_idxes, face_idxes, glasses_idxes) in enumerate(dataloader):
            # Skip the last batch
            if images.size(0) != args.batch_size: continue
            
            # Adversarial ground truths
            valid = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

            # Transfer input to CUDA
            real_imgs = Variable(images.type(torch.cuda.FloatTensor))
            labels = Variable(labels.type(torch.cuda.LongTensor))
            hair_idxes = Variable(hair_idxes.type(torch.cuda.LongTensor))
            eye_idxes = Variable(eye_idxes.type(torch.cuda.LongTensor))
            face_idxes = Variable(face_idxes.type(torch.cuda.LongTensor))
            glasses_idxes = Variable(glasses_idxes.type(torch.cuda.LongTensor))

            # Update the generator
            optimizer_G.zero_grad()
            # Sample noise and labels as generator input
            z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.z_dim))))
            gen_hair_idxes = Variable(torch.cuda.LongTensor(np.random.randint(0, len(datasets.attr_hair), args.batch_size)))
            gen_eye_idxes = Variable(torch.cuda.LongTensor(np.random.randint(0, len(datasets.attr_eye), args.batch_size)))
            gen_face_idxes = Variable(torch.cuda.LongTensor(np.random.randint(0, len(datasets.attr_face), args.batch_size)))
            gen_glasses_idxes = Variable(torch.cuda.LongTensor(np.random.randint(0, len(datasets.attr_glasses), args.batch_size)))
            # Generate a batch of images
            gen_imgs = generator(z, gen_hair_idxes, gen_eye_idxes, gen_face_idxes, gen_glasses_idxes)
            # Loss measures generator's ability to fool the discriminator
            validity, pred_aux_hair, pred_aux_eye, pred_aux_face, pred_aux_glasses = discriminator(gen_imgs)
            g_loss = (adversarial_loss(validity, valid) + 
                (auxiliary_loss(pred_aux_hair, gen_hair_idxes) + 
                auxiliary_loss(pred_aux_eye, gen_eye_idxes) + 
                auxiliary_loss(pred_aux_face, gen_face_idxes) + 
                auxiliary_loss(pred_aux_glasses, gen_glasses_idxes)) / 4) / 2
            g_loss.backward()
            optimizer_G.step()
            
            # Update the discriminator
            for _ in range(args.disc_iter):
                optimizer_D.zero_grad()
                # Loss for real images
                real_pred, real_aux_hair, real_aux_eye, real_aux_face, real_aux_glasses = discriminator(real_imgs)
                d_real_loss = (adversarial_loss(real_pred, valid) + 
                    (auxiliary_loss(real_aux_hair, hair_idxes) + 
                    auxiliary_loss(real_aux_eye, eye_idxes) + 
                    auxiliary_loss(real_aux_face, face_idxes) + 
                    auxiliary_loss(real_aux_glasses, glasses_idxes)) / 4) / 2
                # Loss for fake images
                fake_pred, fake_aux_hair, fake_aux_eye, fake_aux_face, fake_aux_glasses = discriminator(gen_imgs.detach())
                d_fake_loss = (adversarial_loss(fake_pred, fake) + 
                    (auxiliary_loss(fake_aux_hair, gen_hair_idxes) + 
                    auxiliary_loss(fake_aux_eye, gen_eye_idxes) + 
                    auxiliary_loss(fake_aux_face, gen_face_idxes) + 
                    auxiliary_loss(fake_aux_glasses, gen_glasses_idxes)) / 4) / 2
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                # Calculate discriminator accuracy
                pred_hair = np.concatenate([real_aux_hair.data.cpu().numpy(), fake_aux_hair.data.cpu().numpy()], axis=0)
                gt_hair = np.concatenate([hair_idxes.data.cpu().numpy(), gen_hair_idxes.data.cpu().numpy()], axis=0)
                d_acc_hair = np.mean(np.argmax(pred_hair, axis=1) == gt_hair)
                pred_eye = np.concatenate([real_aux_eye.data.cpu().numpy(), fake_aux_eye.data.cpu().numpy()], axis=0)
                gt_eye = np.concatenate([eye_idxes.data.cpu().numpy(), gen_eye_idxes.data.cpu().numpy()], axis=0)
                d_acc_eye = np.mean(np.argmax(pred_eye, axis=1) == gt_eye)
                pred_face = np.concatenate([real_aux_face.data.cpu().numpy(), fake_aux_face.data.cpu().numpy()], axis=0)
                gt_face = np.concatenate([face_idxes.data.cpu().numpy(), gen_face_idxes.data.cpu().numpy()], axis=0)
                d_acc_face = np.mean(np.argmax(pred_face, axis=1) == gt_face)
                pred_glasses = np.concatenate([real_aux_glasses.data.cpu().numpy(), fake_aux_glasses.data.cpu().numpy()], axis=0)
                gt_glasses = np.concatenate([glasses_idxes.data.cpu().numpy(), gen_glasses_idxes.data.cpu().numpy()], axis=0)
                d_acc_glasses = np.mean(np.argmax(pred_glasses, axis=1) == gt_glasses)

            # Collect the training log for a batch
            batch_g_losses.append(g_loss.item())
            batch_d_losses.append(d_loss.item())
            batch_d_accs_hair.append(d_acc_hair)
            batch_d_accs_eye.append(d_acc_eye)
            batch_d_accs_face.append(d_acc_face)
            batch_d_accs_glasses.append(d_acc_glasses)
            batch_g_loss = sum(batch_g_losses) / len(batch_g_losses)
            batch_d_loss = sum(batch_d_losses) / len(batch_d_losses)
            batch_d_acc_hair = sum(batch_d_accs_hair) / len(batch_d_accs_hair)
            batch_d_acc_eye = sum(batch_d_accs_eye) / len(batch_d_accs_eye)
            batch_d_acc_face = sum(batch_d_accs_face) / len(batch_d_accs_face)
            batch_d_acc_glasses = sum(batch_d_accs_glasses) / len(batch_d_accs_glasses)
            batch_time = time.time() - start_time

            print("[Epoch %d/%d][Batch %d/%d][D loss: %.4f, acc: %.4f/%.4f/%.4f/%.4f][G loss: %.4f][elapsed time: %.2fs]" % \
                (epoch, args.epochs, batch_idx, len(dataloader), batch_d_loss, batch_d_acc_hair, batch_d_acc_eye, batch_d_acc_face, batch_d_acc_glasses, batch_g_loss, batch_time), end='\r')
        print()
        
        # Collect the training log for an epoch
        epoch_g_losses.append(batch_g_loss)
        epoch_d_losses.append(batch_d_loss)
        epoch_d_acc_hair.append(batch_d_acc_hair)
        epoch_d_acc_eye.append(batch_d_acc_eye)
        epoch_d_acc_face.append(batch_d_acc_face)
        epoch_d_acc_glasses.append(batch_d_acc_glasses)
        epoch_time.append(batch_time)

        if epoch % args.save_freq == 0:
            print('Saving the last models...')
            torch.save(discriminator.state_dict(), os.path.join(args.ckpt_dir, 'disc_' + args.ckpt_last))
            torch.save(generator.state_dict(), os.path.join(args.ckpt_dir, 'gen_' + args.ckpt_last))
            np.save(os.path.join(args.ckpt_dir, 'epoch_g_losses.npy'), np.array(epoch_g_losses))
            np.save(os.path.join(args.ckpt_dir, 'epoch_d_losses.npy'), np.array(epoch_d_losses))
            np.save(os.path.join(args.ckpt_dir, 'epoch_d_acc_hair.npy'), np.array(epoch_d_acc_hair))
            np.save(os.path.join(args.ckpt_dir, 'epoch_d_acc_eye.npy'), np.array(epoch_d_acc_eye))
            np.save(os.path.join(args.ckpt_dir, 'epoch_d_acc_face.npy'), np.array(epoch_d_acc_face))
            np.save(os.path.join(args.ckpt_dir, 'epoch_d_acc_glasses.npy'), np.array(epoch_d_acc_glasses))
            np.save(os.path.join(args.ckpt_dir, 'epoch_time.npy'), np.array(epoch_time))
        
        if epoch % args.eval_freq == 0:
            print('Evaluating...')
            # Generate evaluation attributes
            eval_hair_idxes = []
            eval_eye_idxes = []
            eval_face_idxes = []
            eval_glasses_idxes = []
            for hair_idx in range(len(datasets.attr_hair)):
                for eye_idx in range(len(datasets.attr_eye)):
                    for face_idx in range(len(datasets.attr_face)):
                        for glasses_idx in range(len(datasets.attr_glasses)):
                            eval_hair_idxes.append(hair_idx)
                            eval_eye_idxes.append(eye_idx)
                            eval_face_idxes.append(face_idx)
                            eval_glasses_idxes.append(glasses_idx)
            # Transfer input to CUDA
            eval_hair_idxes = Variable(torch.cuda.LongTensor(eval_hair_idxes))
            eval_eye_idxes = Variable(torch.cuda.LongTensor(eval_eye_idxes))
            eval_face_idxes = Variable(torch.cuda.LongTensor(eval_face_idxes))
            eval_glasses_idxes = Variable(torch.cuda.LongTensor(eval_glasses_idxes))
            with torch.no_grad():
                gen_imgs = generator(fixed_z, eval_hair_idxes, eval_eye_idxes, eval_face_idxes, eval_glasses_idxes)
            
            # Plot the generated images
            nrow = int(np.sqrt(len(gen_imgs)))
            num_samples = int(nrow ** 2)
            save_image(gen_imgs[:num_samples], 
                args.eval_dir + '/%d.png' % (epoch), 
                nrow=nrow, normalize=True, range=(-1, 1))

def main():
    # Parse command line and run
    args = parse()
    run(args)

if __name__ == '__main__':
    main()