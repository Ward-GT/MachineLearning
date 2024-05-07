from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import re
import torch
from config import *

def extract_dimensions_from_filename(filename):
    # Define the pattern to match dimensions and their numerical values
    pattern = r'(hw1|hw2|dww_ii_x|dww_oo_x|dww_x|lcore_x1_IW|dcs|hw|dw)_([0-9.]+)'
    filename = os.path.splitext(filename)[0]
    # Use re.findall to find all matches of the pattern
    matches = re.findall(pattern, filename)

    # Convert the matches into a dictionary where keys are dimensions and values are numerical values
    dimensions = {dim: float(value) for dim, value in matches}

    return dimensions

def extract_dimensions(input_dir, file_name):
    # List all files in the input directory
    files = os.listdir(input_dir)

    # Initialize an empty dictionary to store the dimensions
    dimensions_dict = {}

    # Process each file
    for i, file in enumerate(files):
        print(file)
        # Extract the dimensions from the file name
        dimensions = extract_dimensions_from_filename(file)

        # Store the dimensions in the dictionary
        dimensions_dict[file] = dimensions

        # Add the index to the dictionary
        dimensions_dict[file]["index"] = i

    return dimensions_dict

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
        dimensions = extract_dimensions_from_filename(self.input_images[idx])
        input_image = read_image(input_image_path)
        label_image = read_image(label_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image, dimensions