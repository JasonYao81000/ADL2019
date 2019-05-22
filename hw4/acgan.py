import torch
from torch import nn

IMAGE_CHANNELS = 3
IMAGE_SIZE = 128

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, z_dim, datasets):
        super(Generator, self).__init__()

        self.hair_emb = nn.Embedding(len(datasets.attr_hair), z_dim // 4)
        self.eye_emb = nn.Embedding(len(datasets.attr_eye), z_dim // 4)
        self.face_emb = nn.Embedding(len(datasets.attr_face), z_dim // 4)
        self.glasses_emb = nn.Embedding(len(datasets.attr_glasses), z_dim // 4)

        self.init_size = IMAGE_SIZE // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, IMAGE_CHANNELS, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, hair_idxes, eye_idxes, face_idxes, glasses_idxes):
        embedding = torch.cat((
            self.hair_emb(hair_idxes), 
            self.eye_emb(eye_idxes), 
            self.face_emb(face_idxes), 
            self.glasses_emb(glasses_idxes)), dim=-1)
        gen_input = torch.mul(embedding, noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, datasets):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(IMAGE_CHANNELS, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = IMAGE_SIZE // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer_hair = nn.Sequential(nn.Linear(128 * ds_size ** 2, len(datasets.attr_hair)), nn.Softmax(dim=-1))
        self.aux_layer_eye = nn.Sequential(nn.Linear(128 * ds_size ** 2, len(datasets.attr_eye)), nn.Softmax(dim=-1))
        self.aux_layer_face = nn.Sequential(nn.Linear(128 * ds_size ** 2, len(datasets.attr_face)), nn.Softmax(dim=-1))
        self.aux_layer_glasses = nn.Sequential(nn.Linear(128 * ds_size ** 2, len(datasets.attr_glasses)), nn.Softmax(dim=-1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        hair = self.aux_layer_hair(out)
        eye = self.aux_layer_eye(out)
        face = self.aux_layer_face(out)
        glasses = self.aux_layer_glasses(out)

        return validity, hair, eye, face, glasses