from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torch
from config import *

class LabeledDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        """
        Args:
            input_dir (string): Directory with all the input images.
            label_dir (string): Directory with all the label images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform

        self.input_images = os.listdir(input_dir)
        self.label_images = os.listdir(label_dir)

        if len(self.input_images) != len(self.label_images):
            raise ValueError("Number of input images and label images do not match")

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        label_image_path = os.path.join(self.label_dir, self.label_images[idx])
        input_image = read_image(input_image_path)
        label_image = read_image(label_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image