import torch
import numpy as np
import torch.nn as nn
from config import *

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image = nn.Sequential(
            nn.ConvTranspose2d(dim_noise,
                               out_channels=256,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False
                               ),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.label = nn.Sequential(
            nn.ConvTranspose2d(num_class,
                               out_channels=256,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False
                               ),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,
                               128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,
                               1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
            ),
            nn.Tanh()                
        )
    def forward(self, noise, label):
        image = self.image(noise)
        label = self.label(label)
        x = torch.cat((image, label), dim=1)
        x = self.main(x)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()   
        self.image = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label = nn.Sequential(
            nn.Conv2d(in_channels=num_class,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False
            ),
            nn.Sigmoid()

        )
    
    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        x = torch.cat((image, label), dim=1)
        x = self.main(x)
        return x 
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
   
