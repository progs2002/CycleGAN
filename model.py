import torch
import torch.nn as nn

class G_up_Block(nn.Module):
    def __init__(self, in_channels, out_channels, k, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, stride=stride ,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            G_up_Block(channels, channels, 3, 1),
            nn.Dropout(),
            nn.ReflectionPad2d(1),
            G_up_Block(channels, channels, 3, 1)
        )
    def forward(self, x):
        return x + self.layers(x)

class G_down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, k, stride=1, padding=0, output_padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, k, stride, padding, output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_residual_blocks):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks
        self.up = nn.Sequential(
            nn.ReflectionPad2d(3),
            G_up_Block(3,64,7,1,0),
            G_up_Block(64,128,3,2,1),
            G_up_Block(128,256,3,2,1)
        )
        self.residual = nn.Sequential(*[Residual_Block(256) for _ in range(self.num_residual_blocks)])
        self.down = nn.Sequential(
            G_down_Block(256,128,3,2,1),
            G_down_Block(128,64,3,2,1),
            nn.ReflectionPad2d(3),
            G_up_Block(64,3,7,1,0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.up(x)
        x = self.residual(x)
        x = self.down(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,64,4,2,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1,bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1,bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,1,1,bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(512,1,4,1,1),
        )
    
    def forward(self, x):
        return self.layers(x)


gen = Generator(1)
test = torch.randn(1,3,256,256)
print(gen(test).shape)