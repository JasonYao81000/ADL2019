# DCGAN-like generator and discriminator
from torch import nn
from torch.nn.utils import spectral_norm

IMAGE_CHANNELS = 3
IMAGE_SIZE = 128

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            spectral_norm(nn.ConvTranspose2d(z_dim, IMAGE_SIZE * 8, 4, stride=1, padding=0, bias=False)),
            nn.ReLU(True),
            # state size. (IMAGE_SIZE * 8) x 4 x 4
            spectral_norm(nn.ConvTranspose2d(IMAGE_SIZE * 8, IMAGE_SIZE * 4, 4, stride=2, padding=1, bias=False)),
            nn.ReLU(True),
            # state size. (IMAGE_SIZE * 4) x 8 x 8
            spectral_norm(nn.ConvTranspose2d(IMAGE_SIZE * 4, IMAGE_SIZE * 2, 4, stride=2, padding=1, bias=False)),
            nn.ReLU(True),
            # state size. (IMAGE_SIZE * 2) x 16 x 16
            spectral_norm( nn.ConvTranspose2d(IMAGE_SIZE * 2, IMAGE_SIZE, 4, stride=2, padding=1, bias=False)),
            nn.ReLU(True),
            # state size. (IMAGE_SIZE) x 32 x 32
            spectral_norm(nn.ConvTranspose2d(IMAGE_SIZE, IMAGE_SIZE // 2, 4, stride=2, padding=1, bias=False)),
            nn.ReLU(True),
            # state size. (IMAGE_SIZE // 2) x 64 x 64
            spectral_norm(nn.ConvTranspose2d(IMAGE_SIZE // 2, IMAGE_CHANNELS, 4, stride=2, padding=1, bias=False)),
            nn.Tanh()
            # state size. (IMAGE_CHANNELS) x 128 x 128
        )
        for m in self.modules():
            weights_init(m)

    def forward(self, z):
        return self.main(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (IMAGE_CHANNELS) x 128 x 128
            spectral_norm(nn.Conv2d(IMAGE_CHANNELS, IMAGE_SIZE, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE) x 128 x 128
            spectral_norm(nn.Conv2d(IMAGE_SIZE, IMAGE_SIZE, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE) x 64 x 64
            spectral_norm(nn.Conv2d(IMAGE_SIZE, IMAGE_SIZE * 2, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE * 2) x 64 x 64
            spectral_norm(nn.Conv2d(IMAGE_SIZE * 2, IMAGE_SIZE * 2, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE * 2) x 32 x 32
            spectral_norm(nn.Conv2d(IMAGE_SIZE * 2, IMAGE_SIZE * 4, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE * 4) x 32 x 32
            spectral_norm(nn.Conv2d(IMAGE_SIZE * 4, IMAGE_SIZE * 4, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (IMAGE_SIZE * 4) x 16 x 16
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(IMAGE_SIZE * 4 * 16 * 16, 1)),
            nn.Sigmoid()
        )

        for m in self.modules():
            weights_init(m)

    def forward(self, image):
        return self.fc(self.main(image).view(-1, IMAGE_SIZE * 4 * 16 * 16))