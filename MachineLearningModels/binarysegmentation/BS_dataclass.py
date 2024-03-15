from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch
from config import *
class SegmentationDataset(Dataset):
    def __init__(self, input_dir, mask_dir, transform=None):
        """
        Args:
            input_dir (string): Directory with all the input images.
            label_dir (string): Directory with all the label images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.input_images = os.listdir(input_dir)
        self.mask_images = os.listdir(mask_dir)

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_image_path)
        input_image = transforms.ToTensor()(input_image).float()

        mask_image_path = os.path.join(self.mask_dir, self.mask_images[idx])
        mask_image = read_image(mask_image_path).float()

        if self.transform:
            input_image = self.transform(input_image)
            mask_image = self.transform(mask_image)

        return input_image, mask_image
def get_data():
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    dataset = SegmentationDataset(IMAGE_DATASET_PATH, MASK_DATASET_PATH, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader
def convert_to_binary_mask(input_dir, output_dir, threshold=128):
    """
    Converts all images in the input directory to binary mask format
    and saves them to the output directory.

    Parameters:
    - input_dir: Path to the directory containing the images to be converted.
    - output_dir: Path to the directory where the binary masks will be saved.
    - threshold: The value above which a pixel will be set to white, and below
                 which it will be set to black.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define a transform to convert images to binary masks
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: (x < threshold / 255.0).to(torch.float)),  # Apply thresholding
    ])

    # List all files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Process each file
    for filename in image_files:
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)

        # Open the image file
        with Image.open(file_path) as img:
            # Apply the transform to the image
            mask_tensor = transform(img)

            # Convert the tensor to PIL image
            mask_image = transforms.ToPILImage()(mask_tensor)

            # Construct the full output file path
            output_file_path = os.path.join(output_dir, filename)

            # Save the binary mask image
            mask_image.save(output_file_path)
            print(f"Saved binary mask to {output_file_path}")