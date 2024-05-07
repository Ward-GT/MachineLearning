import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from SDE_dataclass import LabeledDataset
from config import *

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        images.append(img)
    return images

def sample_model_output(model: torch.nn.Module, sampler, n: int, batch_size: int = BATCH_SIZE, device=DEVICE, test_path: str = None, test_dataloader: torch.utils.data.DataLoader = None):
    if test_dataloader is not None and test_path is None:
        print("Using test dataloader")
        dataloader = test_dataloader
    elif test_path is not None and test_dataloader is None:
        print("Using test data")
        dataloader = get_test_data(test_path, batch_size=batch_size)
    else:
        _, dataloader = get_data(batch_size)

    model = model.to(device)

    references_list = []
    generated_list = []
    structures_list = []
    iterator = iter(dataloader)
    print(f"Sampling on {device}")
    for i in range(0, n, batch_size):
        references, structures, _ = next(iterator)
        structures = structures.to(device)
        references = references.to(device)
        generated, structures = sampler.sample(model, batch_size, structures)
        references = tensor_to_PIL(references)

        references_list.extend(references)
        generated_list.extend(generated)
        structures_list.extend(structures)
        print(f"Reference: {len(references_list)}, Generated: {len(generated_list)}, Structures: {len(structures_list)}")

    generated_list = [convert_grey_to_white(image) for image in generated_list]

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

def get_data(batch_size: int = BATCH_SIZE, split: bool = True):
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, transform=data_transform)

    if split == True:
        train_size = int((1 - TEST_SPLIT - VALIDATION_SPLIT) * len(dataset))
        val_size = int(VALIDATION_SPLIT * len(dataset))
        test_size = int(TEST_SPLIT * len(dataset))

        set_seed()
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        torch.save(train_dataset.indices, os.path.join(RESULT_PATH, "train_indices.pth"))
        torch.save(val_dataset.indice, os.path.join(RESULT_PATH, "val_indices.pth"))
        torch.save(test_dataset.indices, os.path.join(RESULT_PATH, "test_indices.pth"))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader, dataset, -1, -1, -1, -1

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
    print(f"Made subset with {len(test_subset)} images")
    set_seed()
    test_dataloader = DataLoader(test_subset, batch_size=batch_size)

    return test_dataloader

def concatenate_images(images: torch.Tensor, structures: torch.Tensor):
    return torch.cat((images, structures), dim=1)

def split_images(concatenated_images):
    return concatenated_images[:, :3], concatenated_images[:, 3:]

def concat_to_batchsize(images: torch.Tensor, n: int):
    m = images.shape[0]
    if m == n:
        return images
    elif m < n:
        indices = torch.arange(0, n-m)
        return torch.cat((images, images[indices]), dim=0)
    else:
        return images[:n]

def tensor_to_PIL(tensor: torch.Tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    images = []
    for i in range(tensor.shape[0]):
        image = transforms.ToPILImage()(tensor[i])
        images.append(image)
    return images

def convert_grey_to_white(image: Image, threshold: int = 200):
    """
    Converts light gray pixels in the image to white.

    Parameters:
    - image: The Image to be Converted
    - threshold: The value above which a pixel will be set to white.
    """

    image_array = np.array(image)
    # Define a function to apply to each pixel
    def change_color(pixel):
        # If all the channels of the pixel value are above the threshold, change it to white
        if np.all(pixel > threshold):
            return (255, 255, 255)
        else:
            return pixel

    # Apply the function to each pixel in the image
    new_image_array = np.apply_along_axis(change_color, axis=-1, arr=image_array)
    new_image = Image.fromarray(new_image_array.astype('uint8'), 'RGB')
    return new_image

def set_seed(seed: int = DEFAULT_SEED, fully_deterministic: bool = False):
    """
    Set seed for reproducible behavior.

    Parameters
    ----------
    seed : int
        Seed value to set. By default, 1958.
    fully_deterministic : bool
        Whether to set the environment to fully deterministic. By default, False.
        This should only be used for debugging and testing, as it can significantly
        slow down training at little to no benefit.
    """
    if fully_deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed set!")



