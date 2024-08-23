import torch
import torch.nn as nn

class G_down_block(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                padding_mode="reflect"
            )
        )
        
        if norm:
            self.layers.append(nn.InstanceNorm2d(out_channels))
        
        self.layers.append(nn.LeakyReLU(0.2))
        
    def forward(self, x):
        return self.layers(x) 

class G_up_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, act="leaky-relu"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                output_padding=0
            )
        )
        
        self.layers.append(nn.InstanceNorm2d(out_channels))
        
        if act == "tanh":
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            self.layers.append(nn.Dropout(0.5))
        
    def forward(self, x):
        return self.layers(x) 

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            G_down_block(3,64,norm=False),
            G_down_block(64,128),
            G_down_block(128,256),
            G_down_block(256,512),
            G_down_block(512,512),
            G_down_block(512,512),
            G_down_block(512,512),
            G_down_block(512,512,norm=False)
        )
        self.up = nn.Sequential(
            G_up_block(2*512,512,dropout=True),
            G_up_block(2*512,512,dropout=True),
            G_up_block(2*512,512,dropout=True),
            G_up_block(2*512,512),
            G_up_block(2*512,256),
            G_up_block(2*256,128),
            G_up_block(2*128,64),
            G_up_block(2*64,3,act="tanh"),
        )
    def forward(self, x):
        buffer = []
        for net in self.down:
            x = net(x)
            buffer.append(x)
        
        for res, net in zip(buffer[::-1], self.up):
            x = net(torch.cat([x,res], dim=1))
            
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

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.photo_G = Generator()
        self.photo_D = Discriminator()
        
        self.monet_G = Generator()
        self.monet_D = Discriminator()

        self.apply(self._init_weights)

        print(f'Cycle GAN initialized with {self._get_num_params()} parameters')

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
    def _get_num_params(self):
        p_G = sum([p.numel() for p in self.photo_G.parameters()])
        p_D = sum([p.numel() for p in self.photo_D.parameters()])
        m_G = sum([p.numel() for p in self.monet_G.parameters()])
        m_D = sum([p.numel() for p in self.monet_D.parameters()])
        
        return p_G + p_D + m_G + m_D
    
    def photo_to_monet(self, photo):
        self.monet_G.eval()
        with torch.no_grad():
            photo = torch.unsqueeze(photo, 0)
            out = self.monet_G(photo)
        return out
    
    def monet_to_photo(self, monet):
        self.photo_G.eval()
        with torch.no_grad():
            monet = torch.unsqueeze(monet, 0)
            out = self.photo_G(monet)
        return out


gen = Generator()
test = torch.randn(1,3,256,256)
print(gen(test).shape)