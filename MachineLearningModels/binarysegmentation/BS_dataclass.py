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
        input_image = transforms.ToTensor()(input_image)

        mask_image_path = os.path.join(self.mask_dir, self.mask_images[idx])
        mask_image = Image.open(mask_image_path)
        mask_image = transforms.ToTensor()(mask_image)

        if self.transform:
            input_image = self.transform(input_image)
            mask_image = self.transform(mask_image)

        return input_image, mask_image
def get_data():
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ])

    dataset = SegmentationDataset(IMAGE_DATASET_PATH, MASK_DATASET_PATH, transform=data_transform)

    train_size = int((1 - TEST_SPLIT) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader