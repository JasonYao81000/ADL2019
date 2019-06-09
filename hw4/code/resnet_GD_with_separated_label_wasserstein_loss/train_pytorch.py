import argparse
import os
import numpy as np
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch

from resnet_generator import Generator
from resnet_discriminator import Discriminator
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
    
    # Loss weight for gradient penalty
    lambda_gp = 10
    
    if opt.test != True:
        dataset = Dataset(opt.test, opt.data_path)
    else:
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
    
    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, dataset)
    discriminator = Discriminator(dataset)      
    
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir, exist_ok=True)
    
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    
    print(opt.test)
    epn = 530
    if opt.test:
#        for ep in range(1,600):
        generator.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, 'generator_%d.cpt' % epn)))
        generator.eval()
        discriminator.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, 'discriminator_%d.cpt' % epn)))
        discriminator.eval()
               
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=opt.num_workers)
        sample_index = 0
        for i, (labels, hair_id, eye_id, face_id, glasses_id) in enumerate(dataloader):
            batch_size = labels.size(0)
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            labels = Variable(labels.type(LongTensor))
            hair_id = Variable(hair_id.type(torch.cuda.LongTensor))
            eye_id = Variable(eye_id.type(torch.cuda.LongTensor))
            face_id = Variable(face_id.type(torch.cuda.LongTensor))
            glasses_id = Variable(glasses_id.type(torch.cuda.LongTensor))
            
            # Generate a batch of images
            gen_imgs = generator(z, hair_id, eye_id, face_id, glasses_id)
            for img in gen_imgs:
                save_image(img, image_dir + "%d.png" % (sample_index), normalize=True, range=(-1, 1))
                sample_index += 1
        if opt.mode == 'fid':
            with open('FID2.txt', 'a') as f:
                f.writelines('Epoch_%d:   ' % epn)
#             os.system('python ./run_fid.py ' + './test_images_fid/')
        if opt.mode == 'human':
            os.system('python ./sample_test/merge_images.py ' + './test_images/')
#        epn -= 1
        return
    

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) 
    
    
    # The fixed z for evaluation
    fixed_z = Variable(torch.cuda.FloatTensor(
        np.random.normal(0, 1, (6 * 4 * 3 * 2, opt.latent_dim))))
    # fixed_z = Variable(torch.cuda.FloatTensor(
    #     np.random.binomial(n=1, p=0.5, size=(len(datasets.attr_hair) * len(datasets.attr_eye) * len(datasets.attr_face) * len(datasets.attr_glasses), args.z_dim))))

    # Generate evaluation attributes
    eval_hair_idxes = []
    eval_eye_idxes = []
    eval_face_idxes = []
    eval_glasses_idxes = []
    for hair_idx in range(0,6):
        for eye_idx in range(0,4):
            for face_idx in range(0,3):
                for glasses_idx in range(0,2):
                    eval_hair_idxes.append(hair_idx)
                    eval_eye_idxes.append(eye_idx)
                    eval_face_idxes.append(face_idx)
                    eval_glasses_idxes.append(glasses_idx)
    # Transfer evaluation inputs to CUDA
    eval_hair_idxes = Variable(torch.cuda.LongTensor(eval_hair_idxes))
    eval_eye_idxes = Variable(torch.cuda.LongTensor(eval_eye_idxes))
    eval_face_idxes = Variable(torch.cuda.LongTensor(eval_face_idxes))
    eval_glasses_idxes = Variable(torch.cuda.LongTensor(eval_glasses_idxes))
    
    
    def sample_image(n_row, fix_z, epoch):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            gen_imgs = []
            for batch_start in range(0, eval_hair_idxes.shape[0], opt.batch_size):
                batch_end = batch_start + opt.batch_size
                if batch_end > eval_hair_idxes.shape[0]: batch_end = eval_hair_idxes.shape[0]
                batch_gen_imgs = generator(
                    fixed_z[batch_start:batch_end], 
                    eval_hair_idxes[batch_start:batch_end], 
                    eval_eye_idxes[batch_start:batch_end], 
                    eval_face_idxes[batch_start:batch_end], 
                    eval_glasses_idxes[batch_start:batch_end])
                gen_imgs.append(batch_gen_imgs)
            gen_imgs = torch.cat(gen_imgs, dim=0)

        # Plot the generated images
        nrow = int(np.sqrt(len(gen_imgs)))
        num_samples = int(nrow ** 2)
        save_image(gen_imgs[:num_samples], 
            "images/%d.png" % (epoch), 
            nrow=nrow, normalize=True, range=(-1, 1))
        
    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _, _, _, _ = D(interpolates)
        fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    
    # ----------
    #  Training
    # ----------
    
    generator.train()
    discriminator.train()
        
    # Configure data loader
    #os.makedirs("../../data/mnist", exist_ok=True)
    
        
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
    n_row = 12
    fix_z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    for epoch in range(opt.n_epochs):
        start_time = time.time()
        generator.train()
        discriminator.train()
        
        batch_g_losses = []
        batch_d_losses = []
        batch_d_accs_hair = []
        batch_d_accs_eye = []
        batch_d_accs_face = []
        batch_d_accs_glasses = []
        
        for i, (imgs, labels, hair_id, eye_id, face_id, glasses_id) in enumerate(dataloader):
    
            batch_size = imgs.shape[0]
    
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
    
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            hair_id = Variable(hair_id.type(LongTensor))
            eye_id = Variable(eye_id.type(LongTensor))
            face_id = Variable(face_id.type(LongTensor))
            glasses_id = Variable(glasses_id.type(LongTensor))
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
    
            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_hair_id = Variable(LongTensor(np.random.randint(0, 6, batch_size)))
            gen_eye_id = Variable(LongTensor(np.random.randint(0, 4, batch_size)))
            gen_face_id = Variable(LongTensor(np.random.randint(0, 3, batch_size)))
            gen_glasses_id = Variable(LongTensor(np.random.randint(0, 2, batch_size)))
    
            # Generate a batch of images
            gen_imgs = generator(z, gen_hair_id, gen_eye_id, gen_face_id, gen_glasses_id)
#            print(gen_imgs.size())
            # Loss measures generator's ability to fool the discriminator
            validity, pred_aux_hair, pred_aux_eye, pred_aux_face, pred_aux_glasses = discriminator(gen_imgs)
            g_loss = 0.8 * -validity.mean() + 0.2 * (auxiliary_loss(pred_aux_hair, gen_hair_id) + 
                   auxiliary_loss(pred_aux_eye, gen_eye_id) + 
                   auxiliary_loss(pred_aux_face, gen_face_id) + 
                   auxiliary_loss(pred_aux_glasses, gen_glasses_id))
    
            g_loss.backward()
            optimizer_G.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Loss for real images
            real_pred, real_aux_hair, real_aux_eye, real_aux_face, real_aux_glasses = discriminator(real_imgs)
            d_real_loss = 0.8 * -real_pred.mean() + 0.2 * (auxiliary_loss(real_aux_hair, hair_id) + 
                    auxiliary_loss(real_aux_eye, eye_id) + 
                    auxiliary_loss(real_aux_face, face_id) + 
                    auxiliary_loss(real_aux_glasses, glasses_id))
    
            # Loss for fake images
            fake_pred, fake_aux_hair, fake_aux_eye, fake_aux_face, fake_aux_glasses = discriminator(gen_imgs.detach())
            d_fake_loss = 0.8 * fake_pred.mean() + 0.2 * (auxiliary_loss(fake_aux_hair, gen_hair_id) + 
                    auxiliary_loss(fake_aux_eye, gen_eye_id) + 
                    auxiliary_loss(fake_aux_face, gen_face_id) + 
                    auxiliary_loss(fake_aux_glasses, gen_glasses_id))
    
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
    
            # Calculate discriminator accuracy
            pred_hair = np.concatenate([real_aux_hair.data.cpu().numpy(), fake_aux_hair.data.cpu().numpy()], axis=0)
            gt_hair = np.concatenate([hair_id.data.cpu().numpy(), gen_hair_id.data.cpu().numpy()], axis=0)
            d_acc_hair = np.mean(np.argmax(pred_hair, axis=1) == gt_hair)
            pred_eye = np.concatenate([real_aux_eye.data.cpu().numpy(), fake_aux_eye.data.cpu().numpy()], axis=0)
            gt_eye = np.concatenate([eye_id.data.cpu().numpy(), gen_eye_id.data.cpu().numpy()], axis=0)
            d_acc_eye = np.mean(np.argmax(pred_eye, axis=1) == gt_eye)
            pred_face = np.concatenate([real_aux_face.data.cpu().numpy(), fake_aux_face.data.cpu().numpy()], axis=0)
            gt_face = np.concatenate([face_id.data.cpu().numpy(), gen_face_id.data.cpu().numpy()], axis=0)
            d_acc_face = np.mean(np.argmax(pred_face, axis=1) == gt_face)
            pred_glasses = np.concatenate([real_aux_glasses.data.cpu().numpy(), fake_aux_glasses.data.cpu().numpy()], axis=0)
            gt_glasses = np.concatenate([glasses_id.data.cpu().numpy(), gen_glasses_id.data.cpu().numpy()], axis=0)
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
    
            
            batch_time = time.time()-start_time
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, acc: %.4f/%.4f/%.4f/%.4f][G loss: %.4f] [time: %.2fs]"
                % (epoch, opt.n_epochs, i, len(dataloader), batch_d_loss, batch_d_acc_hair, batch_d_acc_eye, batch_d_acc_face, batch_d_acc_glasses, batch_g_loss, batch_time), end="\r"
            )
            batches_done = epoch * len(dataloader) + i
            
#            batch_g_losses.append(g_loss.item())
#            batch_d_losses.append(d_loss.item())
#            batch_d_acc.append(d_acc)
#         print(batch_d_acc_hair)
        a = (batch_d_acc_hair + batch_d_acc_eye + batch_d_acc_face + batch_d_acc_glasses) /4
        g = np.mean(np.array(batch_g_losses))
        d = np.mean(np.array(batch_d_losses))
        
        epoch_time.append(time.time())
        epoch_g_losses.append(g)
        epoch_d_losses.append(d)
        epoch_d_acc.append(a)
        epoch_obvious.append(g*d)
        if epoch % 5 == 0:
            sample_image(n_row=n_row, fix_z=fix_z, epoch=epoch)
        print()
        
#         print(
#                 "[Epoch %d/%d] [D loss: %f, G loss: %f] [acc: %d%%] [GD: %f]"
#                 % (epoch, opt.n_epochs, d, g, 100 * a, g*d), end="\r"
#             )
        
        print()
#        batch_g_losses = []
#        batch_d_losses = []
#        batch_d_acc = []
        
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