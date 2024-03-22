import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics import FrechetInceptionDistance
from DDPM_dataclass import MagneticDataset
from config import *

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data():
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = MagneticDataset(IMAGE_DATASET_PATH, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def calculate_FID(real_images, sampled_images):
    fid = FrechetInceptionDistance()
    real_images = transforms.Lambda(lambda t: t / 255.0)(real_images)
    sampled_images = transforms.Lambda(lambda t: t / 255.0)(sampled_images)

    fid.update(images=real_images, is_real=True)
    fid.update(images=sampled_images, is_real=False)
    return fid.compute()