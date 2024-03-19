import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from diffusion_parameters import *



# Define the transform
data_transform = transforms.Compose([
    transforms.Resize((IMG_size, IMG_size)),
    transforms.Lambda(lambda t: t/ 255.0),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def get_image_size(output_dir, structure_dir):
    """
    Returns the minimum size of the images in the output and structure folders.

    Args:
    output_dir (str): The path to the folder containing the output images.
    structure_dir (str): The path to the folder containing the structure images.
    """

    # Load the first image from the input and label folders
    output_image = Image.open(os.path.join(output_dir, "0.png"))
    structure_image = Image.open(os.path.join(structure_dir, "0.png"))

    print(f"Structure image size: {structure_image.size}")
    print(f"Output image size: {output_image.size}")

    # Define the size of the images
    image_size = min(structure_image.size, output_image.size)
    return image_size

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

rename_images_to_index(r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B\Output")
rename_images_to_index(r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B\Structure")