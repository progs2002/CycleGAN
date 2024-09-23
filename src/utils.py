from glob import glob
from PIL import Image
import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        
        self.files = glob(f'{img_dir}/*.jpg')
        self.transform = transform
        
    def __len__(self):
        return len(self.files) 
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.files[idx])
        
        if self.transform:
            img = self.transform(img)
            
        return img

class CombinedLoader:
    def __init__(self, photo_ds, monet_ds, batch_size, num_workers):
        self.photo_ds = photo_ds
        self.monet_ds = monet_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.len = max(len(photo_ds), len(monet_ds))//batch_size

    def __iter__(self):
        self.photo_loader = DataLoader(self.photo_ds, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=True)
        self.monet_loader = DataLoader(self.monet_ds, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers, shuffle=True)
        self.photo_iter = itertools.cycle(self.photo_loader)
        self.monet_iter = itertools.cycle(self.monet_loader)
        self.counter = 0
        return self
    
    def __len__(self):
        return self.len

    def __next__(self):
        if self.counter > self.len:
            raise StopIteration
        
        self.counter += 1
        return next(self.photo_iter), next(self.monet_iter)   

class ImageBuffer:
    def __init__(self, size=70):
        self.size = size
        self.buffer = []
    def pass_images(self, imgs):
        out_buffer = []
        for img in imgs:
            if len(self.buffer) < self.size:
                self.buffer.append(img)
                out_buffer.append(img)
            else:
                #faster than random.choice https://stackoverflow.com/questions/6824681/get-a-random-boolean-in-python
                if bool(random.getrandbits(1)): 
                    fetch_img_idx = random.choice(range(self.size))
                    out_buffer.append(self.buffer[fetch_img_idx])
                    self.buffer[fetch_img_idx] = img
                else:
                    out_buffer.append(img)
        return torch.stack(out_buffer, dim=0)

default_train_transform = T.Compose(
    [
        T.Resize((268,268)),
        T.RandomCrop((256,256)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
)

default_inference_transform = T.Compose(
    [
        T.Resize((256,256)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
)


def inverse_transform(img):
    return (img * 0.5) + 0.5