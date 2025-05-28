from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import re
import torch

def extract_dimensions_from_filename(filename):
    """
    Read the dimension parameters from the filename.
    Args:
        filename: str, name of the file to be extracted.

    Returns:
        dimensions_dict: dictionary with keys the names and their associated values.
        dimensions_dict: tensor with only the values.

    """
    # Define the pattern to match dimensions and their numerical values
    pattern = r'(hw1|hw2|dww_ii_x|dww_oo_x|dww_x|lcore_x1_IW|dcs|hw|dw)_([0-9.]+)'
    filename = os.path.splitext(filename)[0]

    # Use re.findall to find all matches of the pattern
    matches = re.findall(pattern, filename)

    # Convert the matches into a dictionary where keys are dimensions and values are numerical values
    dimensions_dict = {dim: float(value) for dim, value in matches}

    return dimensions_dict

class LabeledDataset(Dataset):
    def __init__(self, input_dir: str, label_dir: str, transform=None):
        """
        Args:
            input_dir (string): Directory with all the input images.
            label_dir (string): Directory with all the label images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform

        # Names of files are equal so this ensures equal sorting
        self.input_images = sorted(os.listdir(input_dir))
        self.label_images = self.input_images # os.listdir(label_dir)

        # Calculate beforehand so that it does not need to happen during each batch fetch
        self.dimension_dicts = []
        for image in self.input_images:
            dimension_dict= extract_dimensions_from_filename(image)
            self.dimension_dicts.append(dimension_dict)

        if len(self.input_images) != len(self.label_images):
            raise ValueError("Number of input images and label images do not match")

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        label_image_path = os.path.join(self.label_dir, self.label_images[idx])
        dimensions_dict = self.dimension_dicts[idx]
        input_image = read_image(input_image_path)
        label_image = read_image(label_image_path)

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image, dimensions_dict
