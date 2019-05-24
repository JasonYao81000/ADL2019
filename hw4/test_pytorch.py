import argparse
import os
import numpy as np


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch

from models import Generator, Discriminator
from loaddata import Dataset 


os.makedirs("images", exist_ok=True)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./selected_cartoonset100k/', help="data path")
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/', help="ckpt path")
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)
    
    cuda = True if torch.cuda.is_available() else False
    
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.BCELoss()
    
    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor 
    
    
    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        
        hair = np.random.randint(0, 6, batch_size)
        eyes = np.random.randint(6, 10, batch_size)
        face = np.random.randint(10, 13, batch_size)
        glasses = np.random.randint(13, 15, batch_size)
        
        gen_labels = np.zeros((batch_size,15))
        gen_labels[np.arange(batch_size), hair] = 1
        gen_labels[np.arange(batch_size), eyes] = 1
        gen_labels[np.arange(batch_size), face] = 1
        gen_labels[np.arange(batch_size), glasses] = 1
        gen_labels = Variable(FloatTensor(gen_labels))
        gen_imgs = generator(z, gen_labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
        
    
    # ----------
    #  Training
    # ----------


    # Configure data loader
    #os.makedirs("../../data/mnist", exist_ok=True)
    dataset = Dataset(opt.data_path)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers)
    
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
    
            batch_size = imgs.shape[0]
    
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
    
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
    
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            hair = np.random.randint(0, 6, batch_size)
            eyes = np.random.randint(6, 10, batch_size)
            face = np.random.randint(10, 13, batch_size)
            glasses = np.random.randint(13, 15, batch_size)
            
            gen_labels = np.zeros((batch_size,15))
            gen_labels[np.arange(batch_size), hair] = 1
            gen_labels[np.arange(batch_size), eyes] = 1
            gen_labels[np.arange(batch_size), face] = 1
            gen_labels[np.arange(batch_size), glasses] = 1
            gen_labels = Variable(FloatTensor(gen_labels))
    
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
    
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
    
            g_loss.backward()
            optimizer_G.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
    
            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
    
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
    
            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    
            d_loss.backward()
            optimizer_D.step()
    
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=8, batches_done=batches_done)
                
        torch.save(generator, opt.ckpt_dir + 'generator.cpt')
        torch.save(discriminator, opt.ckpt_dir + 'discriminator.cpt')


if __name__ == '__main__':
    main()