import torch 
import torch.nn as nn 

class Conv_Block(nn.Module):
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
            Conv_Block(channels, channels, 3, 1, 1),
            Conv_Block(channels, channels, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.layers(x)

class Conv_Transpose_Block(nn.Module):
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
            Conv_Block(3,64,7,1,3),
            Conv_Block(64,128,3,2,1),
            Conv_Block(128,256,3,2,1)
        )
        self.residual = nn.Sequential(*[Residual_Block(256) for _ in range(self.num_residual_blocks)])
        self.down = nn.Sequential(
            Conv_Transpose_Block(256,128,3,2,1,1),
            Conv_Transpose_Block(128,64,3,2,1,1),
            Conv_Block(64,3,7,1,3)
        )
    def forward(self, x):
        x = self.up(x)
        x = self.residual(x)
        x = self.down(x)
        return x

class Discriminator_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1, norm=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding),
            nn.InstanceNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Discriminator_Block(3,64,2,1,norm=False),
            Discriminator_Block(64,128,2,1),
            Discriminator_Block(128,256,2,1),
            Discriminator_Block(256,512,1,1),
            Discriminator_Block(512,1,1,1),
        )
    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x

def validate(a):
    D = Discriminator()
    G = Generator(9)
    print(D(a).shape)
    print(G(a).shape)

validate(torch.randn(1,3,256,256))