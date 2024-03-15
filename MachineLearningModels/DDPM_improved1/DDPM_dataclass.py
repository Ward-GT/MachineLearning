from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torch
from config import *
class MagneticDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        """
        Args:
            input_dir (string): Directory with all the input images.
            label_dir (string): Directory with all the label images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.transform = transform

        self.input_images = os.listdir(input_dir)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = read_image(input_image_path)

        if self.transform:
            input_image = self.transform(input_image)

        return input_image