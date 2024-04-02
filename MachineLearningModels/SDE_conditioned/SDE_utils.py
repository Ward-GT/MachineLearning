import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from SDE_dataclass import LabeledDataset
from config import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        images.append(img)
    return images

def mse(imageA, imageB):
    # The 'mean' function calculates the average of the array elements.
    # The 'square' function calculates the squared value of each element.
    # np.subtract(imageA, imageB) computes the difference between the images.
    err = np.sqrt((np.square(np.subtract(imageA, imageB)))/(255**2))
    mean_err = np.mean(err)
    max_err = np.max(err)
    return mean_err, max_err

def calculate_metrics(image_set1, image_set2):
    if len(image_set1) != len(image_set2):
        raise ValueError("Number of images in image sets do not match")

    ssim_values = []
    psnr_values = []
    mse_mean_values = []
    mse_max_values = []

    for i in range(len(image_set1)):
        ssim_values.append(ssim(np.array(image_set1[i]), np.array(image_set2[i]), channel_axis=-1, multichannel=True))
        psnr_values.append(psnr(np.array(image_set1[i]), np.array(image_set2[i]), data_range=np.array(image_set1[i]).max() - np.array(image_set1[i]).min()))
        mse_mean, mse_max = mse(np.array(image_set1[i]).flatten(), np.array(image_set2[i]).flatten())
        mse_mean_values.append(mse_mean)
        mse_max_values.append(mse_max)

    return ssim_values, psnr_values, mse_mean_values, mse_max_values

def sample_model_output(model, sampler, n, batch_size=BATCH_SIZE, device=DEVICE, test_path=None):
    if test_path is not None:
        print("Using test data")
        dataloader = get_test_data(test_path, batch_size=batch_size)
    else:
        _, dataloader = get_data(batch_size)

    model = model.to(device)

    references_list = []
    generated_list = []
    structures_list = []

    for i in range(0, n, batch_size):
        references, structures = next(iter(dataloader))
        structures = structures.to(device)
        references = references.to(device)
        generated, structures = sampler.sample(model, batch_size, structures)
        references = tensor_to_PIL(references)

        references_list.extend(references)
        generated_list.extend(generated)
        structures_list.extend(structures)
        print(f"Reference: {len(references_list)}, Generated: {len(generated_list)}, Structures: {len(structures_list)}")

    return references_list, generated_list, structures_list

def save_image_list(image_list, path):
    for i, image in enumerate(image_list):
        image.save(os.path.join(path, f"{i}.jpg"))

def save_images(reference_images=None, generated_images=None, structure_images=None, path=None, **kwargs):
    # Determine how many image sets are provided
    image_sets_with_titles = {
        'Reference': reference_images,
        'Generated': generated_images,
        'Structure': structure_images
    }

    image_sets = [(title, images) for title, images in image_sets_with_titles.items() if images is not None]
    # Calculate the maximum number of images in any set to set the number of columns
    n_rows = len(image_sets)
    n_cols = max(len(images) for _, images in image_sets)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    for row, (title, image_set) in enumerate(image_sets):
        for col, img in enumerate(image_set):
            axs[row, col].imshow(img)
            axs[row, col].axis('off')  # Hide the axes ticks
            if col == 0:
                axs[row, col].set_title(title, fontweight='bold', size=20)  # Set the title of the row

            # Fill remaining columns of the row with blank axes if any
            for extra_col in range(col + 1, n_cols):
                axs[row, extra_col].axis('off')

    plt.tight_layout()
    if path:
        plt.savefig(path, **kwargs)  # Save the figure if a path is provided

def get_data(batch_size=BATCH_SIZE):
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, transform=data_transform)

    train_size = int((1 - TEST_SPLIT) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    torch.save(train_dataset.indices, os.path.join(RESULT_PATH, "train_indices.pth"))
    torch.save(test_dataset.indices, os.path.join(RESULT_PATH, "test_indices.pth"))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_test_data(test_path, batch_size=BATCH_SIZE):
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, transform=data_transform)

    test_indices = torch.load(test_path)
    print(f"Loaded {len(test_indices)} test indices")
    test_subset = Subset(dataset, test_indices)

    test_dataloader = DataLoader(dataset, batch_size=batch_size)

    return test_dataloader

def concatenate_images(images, structures):
    return torch.cat((images, structures), dim=1)

def split_images(concatenated_images):
    return concatenated_images[:, :3], concatenated_images[:, 3:]

def concat_to_batchsize(images, n):
    m = images.shape[0]
    if m == n:
        return images
    elif m < n:
        indices = torch.arange(0, n-m)
        return torch.cat((images, images[indices]), dim=0)
    else:
        return images[:n]

def tensor_to_PIL(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    images = []
    for i in range(tensor.shape[0]):
        image = transforms.ToPILImage()(tensor[i])
        images.append(image)
    return images

