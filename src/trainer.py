from dataclasses import dataclass
from typing import Tuple

import math
import random
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid

from src.utils import (
    CombinedLoader, 
    ImageBuffer,
    inverse_transform
)

@dataclass
class TrainerConfig:
    log_dir: str
    batch_size: int = 16
    num_workers: int = 2
    device: str|None = None

    h_lambda: int = 10
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
        
    weight_decay: float = 0
    warmup_iters: int = 500
    min_lr: float = 1e-5
    lr_decay_iters: int = 4000

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            

class Trainer:
    def __init__(self, config, photo_ds, monet_ds, model):
        self.model = model
        self.device = config.device

        self.photo_ds = photo_ds
        self.monet_ds = monet_ds
        
        self.lr = config.lr
        
        self.warmup_iters = config.warmup_iters
        self.min_lr = config.min_lr
        self.lr_decay_iters = config.lr_decay_iters

        self.h_lambda = config.h_lambda
        
        self.g_optim, self.d_optim = self._get_optimizers(config)
        
        self.log_writer = SummaryWriter(config.log_dir)
        
        self.loader = self._get_dataloaders(config)
        
        self.fake_monet_buffer = ImageBuffer()
        self.fake_photo_buffer = ImageBuffer()

    #custom lr scheduler inspired by https://github.com/karpathy/nanoGPT
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.lr * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)
        
    def _get_optimizers(self, config):
        g_optim = Adam(
            itertools.chain(
                self.model.photo_G.parameters(), 
                self.model.monet_G.parameters()
            ),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        d_optim = Adam(
            itertools.chain(
                self.model.photo_D.parameters(), 
                self.model.monet_D.parameters()
            ),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        return g_optim, d_optim
    
    def _get_dataloaders(self, config):
        return CombinedLoader(self.photo_ds, self.monet_ds, batch_size=config.batch_size, num_workers=config.num_workers)
       
    def _set_discriminator_requires_grad(self, requires_grad=False):
        for net in [self.model.photo_D, self.model.monet_D]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def _get_gan_loss(self, x, is_real):
        target = torch.full(x.shape, float(is_real), device=self.device)
        return F.mse_loss(x, target)
    
    def _get_discriminator_loss(self, net, real, fake):
        loss_real = self._get_gan_loss(net(real), is_real=True)
        loss_fake = self._get_gan_loss(net(fake), is_real=False)
        loss = (loss_real + loss_fake) * 0.5
        loss.backward()
        return loss
    
    def train_step(self, real_photo, real_monet):
        self.model.train()
        
        fake_photo = self.model.photo_G(real_monet)
        fake_monet = self.model.monet_G(real_photo)
        cycle_photo = self.model.photo_G(fake_monet)
        cycle_monet = self.model.monet_G(fake_photo)
        
        #lock discriminators
        self._set_discriminator_requires_grad(False)
        
        #generator optimization
        self.g_optim.zero_grad()
        
        #identity loss
        id_loss_1 = F.l1_loss(self.model.photo_G(real_photo), real_photo) * 0.5 * self.h_lambda
        id_loss_2 = F.l1_loss(self.model.monet_G(real_monet), real_monet) * 0.5 * self.h_lambda
        id_loss = id_loss_1 + id_loss_2
        
        #generator loss
        photo_G_loss = self._get_gan_loss(
            self.model.photo_D(fake_photo),
            is_real=True
        )
        monet_G_loss = self._get_gan_loss(
            self.model.monet_D(fake_monet),
            is_real=True
        )
        gen_loss = photo_G_loss + monet_G_loss
        
        #cycle loss
        loss_photo_cycle = F.l1_loss(cycle_photo, real_photo) * self.h_lambda
        loss_monet_cycle = F.l1_loss(cycle_monet, real_monet) * self.h_lambda
        cycle_loss = loss_photo_cycle + loss_monet_cycle
        
        #backprop for generators
        total_G_loss =  id_loss + gen_loss + cycle_loss
        total_G_loss.backward()
        
        nn.utils.clip_grad_norm_(
            itertools.chain(
                self.model.photo_G.parameters(), 
                self.model.monet_G.parameters()
            ), 
            1.0
        )
        
        self.g_optim.step()
        
        #unfreeze discriminators
        self._set_discriminator_requires_grad(True)
        
        #discriminator optimization
        self.d_optim.zero_grad()
        
        photo_D_loss = self._get_discriminator_loss(
            self.model.photo_D,
            real_photo,
            self.fake_photo_buffer.pass_images(fake_photo.detach())
        )
        monet_D_loss = self._get_discriminator_loss(
            self.model.monet_D,
            real_monet,
            self.fake_monet_buffer.pass_images(fake_monet.detach())
        )
        
        nn.utils.clip_grad_norm_(
            itertools.chain(
                self.model.photo_D.parameters(), 
                self.model.monet_D.parameters()
            ), 
            1.0
        )
        
        self.d_optim.step()

        return {
            "monet_g_loss": monet_G_loss.item(),
            "photo_g_loss": photo_G_loss.item(),
            "monet_d_loss": monet_D_loss.item(),
            "photo_d_loss": photo_D_loss.item(),
            "identity_loss": id_loss.item(),
            "cyclic_loss": cycle_loss.item(),
            "total_g_loss": total_G_loss.item(),
            "total_d_loss": (photo_D_loss + monet_D_loss).item()
        }
    
    def eval_step(self, step):
        eval_photo = random.choice(self.photo_ds).to(self.device)
        gen_monet = self.model.photo_to_monet(eval_photo.unsqueeze(0)).squeeze(0)
        
        photo2monet_grid = inverse_transform(make_grid([eval_photo, gen_monet]))

        self.log_writer.add_image('photo_to_monet', photo2monet_grid, step)
    
    def train(self, epochs):
        step = 0
        tqdm_g_loss = 0.0  
        tqdm_d_loss = 0.0
        for epoch in range(epochs):
            running_total_g_loss = 0.0
            running_total_d_loss = 0.0
            for photo, monet in (pbar := tqdm(self.loader)):
                pbar.set_description(f"Epoch{epoch:2d} G_loss:{tqdm_g_loss:5.3f} D_loss:{tqdm_d_loss:5.3f}")
                
                photo, monet = photo.to(self.device), monet.to(self.device)
                
                new_lr = self.get_lr(step)
                for param_group in itertools.chain(
                    self.d_optim.param_groups, 
                    self.g_optim.param_groups
                ):
                    param_group['lr'] = new_lr

                metrics_dict = self.train_step(photo, monet)
                
                tqdm_g_loss = metrics_dict["total_g_loss"]
                tqdm_d_loss = metrics_dict["total_d_loss"]

                running_total_g_loss += tqdm_g_loss
                running_total_d_loss += tqdm_d_loss

                for metric in metrics_dict.keys():
                    self.log_writer.add_scalar(str(metric), metrics_dict[metric], step)
                
                step += 1

            # cli log 
            print(f"g_loss: {running_total_g_loss/len(self.loader)} d_loss: {running_total_d_loss/len(self.loader)}")

            # image log
            self.eval_step(step)
