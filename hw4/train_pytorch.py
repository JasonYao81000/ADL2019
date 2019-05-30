import argparse
import os
import numpy as np
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch

from models import Generator, Discriminator
from loaddata import Dataset 

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

os.makedirs("images", exist_ok=True)
os.makedirs("test_images", exist_ok=True)
os.makedirs("test_images_fid", exist_ok=True)
    
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
    parser.add_argument("--test_path", type=str, default='./sample_test/', help="test path")
    parser.add_argument("--mode", choices=['fid', 'human'], default='fid', help="test choices")
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/', help="ckpt path")
    parser.add_argument("--test", action='store_true', default=False, help="test")
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers")
    parser.add_argument("--n_epochs", type=int, default=300000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
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
    
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)      
    
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir, exist_ok=True)
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.BCELoss()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    
    print(opt.test)
    if opt.test:
        generator.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, 'generator.cpt')))
        generator.eval()
        discriminator.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, 'discriminator.cpt')))
        discriminator.eval()
        if opt.mode == 'human':
            test_path = opt.test_path + 'sample_human_testing_labels.txt'
            image_dir = "./test_images/"
        elif opt.mode == 'fid':
            test_path = opt.test_path + 'sample_fid_testing_labels.txt'
            image_dir = "./test_images_fid/"
        else:
            print('no mode is chosen')
            return
        dataset = Dataset(opt.test, test_path)       
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=opt.num_workers)
        sample_index = 0
        for i, labels in enumerate(dataloader):
            batch_size = labels.size(0)
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            labels = Variable(labels.type(torch.cuda.FloatTensor))
            # Generate a batch of images
            gen_imgs = generator(z, labels)
            for img in gen_imgs:
                save_image(img, image_dir + "%d.png" % (sample_index), normalize=True, range=(-1, 1))
                sample_index += 1
        if opt.mode == 'human':
            os.system('python ./sample_test/merge_images.py ' + './test_images/')
        return
    

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) 
    
    
    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        batch_size = n_row ** 2
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        
        eval_hair_idxes = []
        eval_eye_idxes = []
        eval_face_idxes = []
        eval_glasses_idxes = []
        for hair_idx in range(0,6):
            for eye_idx in range(6,10):
                for face_idx in range(10,13):
                    for glasses_idx in range(13,15):
                        eval_hair_idxes.append(hair_idx)
                        eval_eye_idxes.append(eye_idx)
                        eval_face_idxes.append(face_idx)
                        eval_glasses_idxes.append(glasses_idx)
        
        gen_labels = np.zeros((batch_size,15))
        gen_labels[np.arange(batch_size), eval_hair_idxes] = 1
        gen_labels[np.arange(batch_size), eval_eye_idxes] = 1
        gen_labels[np.arange(batch_size), eval_face_idxes] = 1
        gen_labels[np.arange(batch_size), eval_glasses_idxes] = 1
        gen_labels = Variable(FloatTensor(gen_labels))
        gen_imgs = generator(z, gen_labels)
        save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
        gen_labels = gen_labels.tolist()
        with open('./images/' + '%d.txt' % batches_done, 'w') as f:
            for line in gen_labels:
                s = [str(i) for i in line]  
                res = " ".join(s)
                f.writelines(res+'\n')
        
    
    # ----------
    #  Training
    # ----------
    
    generator.train()
    discriminator.train()
        
    # Configure data loader
    #os.makedirs("../../data/mnist", exist_ok=True)
    dataset = Dataset(opt.test, opt.data_path)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers)
    
    epoch_g_losses = []
    epoch_d_losses = []
    epoch_d_acc = []
    epoch_time = []
    epoch_obvious = []
    record_acc = -1
    record_gd = 1000
    save_m = 1
    for epoch in range(opt.n_epochs):
        start_time = time.time()
        batch_g_losses = []
        batch_d_losses = []
        batch_d_acc = []
        
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
#            print(pred)
#            print(gt)
            acc_hair = np.argmax(pred[:,0:6], axis=1) == np.argmax(gt[:,0:6], axis=1)
            acc_eyes = np.argmax(pred[:,6:10], axis=1) == np.argmax(gt[:,6:10], axis=1)
            acc_face = np.argmax(pred[:,10:13], axis=1) == np.argmax(gt[:,10:13], axis=1)
            acc_glasses = np.argmax(pred[:,13:15], axis=1) == np.argmax(gt[:,13:15], axis=1)
            d_acc = (np.mean(acc_hair) + np.mean(acc_eyes) + np.mean(acc_face) + np.mean(acc_glasses))/4
    
            d_loss.backward()
            optimizer_D.step()
            
            batch_time = time.time()-start_time
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, G loss: %f] [acc: %d%%] [time: %.2fs]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), 100 * d_acc, batch_time), end="\r"
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=12, batches_done=batches_done)
            batch_g_losses.append(g_loss.item())
            batch_d_losses.append(d_loss.item())
            batch_d_acc.append(d_acc)
        
        a = np.mean(np.array(batch_d_acc))
        g = np.mean(np.array(batch_g_losses))
        d = np.mean(np.array(batch_d_losses))
        
        epoch_time.append(time.time())
        epoch_g_losses.append(g)
        epoch_d_losses.append(d)
        epoch_d_acc.append(a)
        epoch_obvious.append(g*d)
        print()
        
        print(
                "[Epoch %d/%d] [D loss: %f, G loss: %f] [acc: %d%%] [GD: %f]"
                % (epoch, opt.n_epochs, d, g, 100 * a, g*d), end="\r"
            )
        
        print()
        batch_g_losses = []
        batch_d_losses = []
        batch_d_acc = []
        
#        if record_acc <= a:
#            torch.save(generator.state_dict(), opt.ckpt_dir + 'generator_acc.cpt')
#            torch.save(discriminator.state_dict(), opt.ckpt_dir + 'discriminator_acc.cpt')
#            print("save_accmodel")
#            print()
#            record_acc = a
        torch.save(generator.state_dict(), opt.ckpt_dir + 'generator_%d.cpt' % save_m)
        torch.save(discriminator.state_dict(), opt.ckpt_dir + 'discriminator_%d.cpt' % save_m)
        print("save_gdmodel")
        print()
        save_m += 1
#            record_gd = g*d
        
#        torch.save(generator.state_dict(), opt.ckpt_dir + 'generator.cpt')
#        torch.save(discriminator.state_dict(), opt.ckpt_dir + 'discriminator.cpt')
            
        np.save(os.path.join(opt.ckpt_dir, 'epoch_g_losses.npy'), np.array(epoch_g_losses))
        np.save(os.path.join(opt.ckpt_dir, 'epoch_d_losses.npy'), np.array(epoch_d_losses))
        np.save(os.path.join(opt.ckpt_dir, 'epoch_d_acc.npy'), np.array(epoch_d_acc))
        np.save(os.path.join(opt.ckpt_dir, 'epoch_time.npy'), np.array(epoch_time))
        


if __name__ == '__main__':
    main()