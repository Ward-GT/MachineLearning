import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics import FrechetInceptionDistance
from SDE_dataclass import LabeledDataset
from config import *

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
def save_images_structures(images, structures, path, **kwargs):
    grid = torchvision.utils.make_grid(torch.cat((images, structures), dim=2), **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data():
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def concatenate_images(images, structures):
    return torch.cat((images, structures), dim=1)

def split_images(concatenated_images):
    return concatenated_images[:, :3], concatenated_images[:, 3:]

def concat_to_batchsize(images, n):
    m = images.shape[0]
    if m == n:
        return images
    elif m < n:
        indices = torch.randint(0, m, (n-m,))
        return torch.cat((images, images[indices]), dim=0)
    else:
        indices = torch.randint(0, m, (n,))
        return torch.cat([images[i].unsqueeze(0) for i in indices], dim=0)

def tensor_to_image(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    return tensor

