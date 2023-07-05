import torch 
import torch.nn as nn 

class Layer_c7s1(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.LazyConv2d(k,(7,7),stride=1,padding=3)
        self.instance_norm = nn.InstanceNorm2d(k)
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = torch.relu(x)
        return x

class Layer_d(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.LazyConv2d(k,(3,3),stride=2,padding=1)
        self.instance_norm = nn.InstanceNorm2d(k)
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = torch.relu(x)
        return x

class Layer_R(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = nn.LazyConv2d(k,(3,3),padding=1)
        self.instance_norm = nn.InstanceNorm2d(k)
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.instance_norm(x)
        x = torch.relu(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = torch.relu(x)
        x += residual
        return x

class Layer_u(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.conv = nn.LazyConvTranspose2d(k,(3,3),stride=2,padding=1,output_padding=1)
        self.instance_norm = nn.InstanceNorm2d(k)
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = torch.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_residual_blocks):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks
        self.layers1 = nn.Sequential(
            Layer_c7s1(64),
            Layer_d(128),
            Layer_d(256)
        )
        self.residual = Layer_R(256)
        self.layers2 = nn.Sequential(
            Layer_u(128),
            Layer_u(64),
            Layer_c7s1(3)
        )
    def forward(self, x):
        x = self.layers1(x)
        for _ in range(self.num_residual_blocks):
            x = self.residual(x)
        x = self.layers2(x)
        return x

class Layer_C(nn.Module):
    def __init__(self, k, stride=2, padding=0, norm=True):
        super().__init__()
        self.conv = nn.LazyConv2d(k,(4,4),stride=stride,padding=padding)
        if norm:
            self.instance_norm = nn.InstanceNorm2d(k)
        else:
            self.instance_norm = nn.Identity()
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.act(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Layer_C(64,2,1,norm=False),
            Layer_C(128,2,1),
            Layer_C(256,2,1),
            Layer_C(512,1,1),
            Layer_C(1,1,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x