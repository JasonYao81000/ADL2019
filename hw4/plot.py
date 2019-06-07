import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = './results/'
if not os.path.exists(results_dir):
        os.makedirs(results_dir)

def plot_losses(ckpt_dir, results_dir, arch):
    epoch_d_losses = np.load(ckpt_dir + 'epoch_d_losses.npy')
    epoch_g_losses = np.load(ckpt_dir + 'epoch_g_losses.npy')
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(epoch_d_losses, label='Discriminator Loss')
    plt.plot(epoch_g_losses, label='Generator Loss')
    plt.legend(loc='center right')
    plt.title('Loss v.s. Epoch [' + arch + ']')
    plt.ylabel('Loss')
    plt.xlabel('# of epochs')
    plt.savefig(results_dir + arch + '_losses.png')

def plot_accs(ckpt_dir, results_dir, arch):
    epoch_d_acc_hair = np.load(ckpt_dir + 'epoch_d_acc_hair.npy')
    epoch_d_acc_eye = np.load(ckpt_dir + 'epoch_d_acc_eye.npy')
    epoch_d_acc_face = np.load(ckpt_dir + 'epoch_d_acc_face.npy')
    epoch_d_acc_glasses = np.load(ckpt_dir + 'epoch_d_acc_glasses.npy')
    plt.clf()
    plt.figure(figsize=(16, 5))
    plt.plot(epoch_d_acc_hair, label='Hair')
    plt.plot(epoch_d_acc_eye, label='Eye')
    plt.plot(epoch_d_acc_face, label='Face')
    plt.plot(epoch_d_acc_glasses, label='Glasses')
    plt.legend(loc='lower right')
    plt.title('Accuracy v.s. Epoch [' + arch + ']')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epochs')
    plt.ylim((0.8, 1.0))
    plt.savefig(results_dir + arch + '_accs.png')

plot_losses('./checkpoints/acgan_500/', results_dir, 'acgan_500')
plot_accs('./checkpoints/acgan_500/', results_dir, 'acgan_500')
plot_losses('./checkpoints/resnet_500/', results_dir, 'resnet_500')
plot_accs('./checkpoints/resnet_500/', results_dir, 'resnet_500')
plot_losses('./checkpoints/resnet_hinge_500/', results_dir, 'resnet_hinge_500')
plot_accs('./checkpoints/resnet_hinge_500/', results_dir, 'resnet_hinge_500')
plot_losses('./checkpoints/resnet_1000/', results_dir, 'resnet_1000')
plot_accs('./checkpoints/resnet_1000/', results_dir, 'resnet_1000')