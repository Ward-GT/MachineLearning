import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torcheval.metrics import FrechetInceptionDistance
from SDE_dataclass import LabeledDataset
from config import *

def save_model_output(model, sampler, img_size=IMAGE_SIZE,  n=BATCH_SIZE, device=DEVICE, path=None):
    if path is not None:
        path = os.path.join(path, "output.jpg")

    dataloader = get_data(n)
    references, structures = next(iter(dataloader))
    model = model.to(device)
    structures = structures.to(device)
    references = references.to(device)
    generated, structures = sampler.sample(model, n, structures)
    references = tensor_to_PIL(references)
    save_images(reference_images=references, generated_images=generated, structure_images=structures, path=path)

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
    plt.show()  # Show the plot

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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader

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

def tensor_to_PIL(tensor):
    tensor = (tensor.clamp(-1, 1) + 1) / 2
    tensor = (tensor * 255).type(torch.uint8)
    images = []
    for i in range(tensor.shape[0]):
        image = transforms.ToPILImage()(tensor[i])
        images.append(image)
    return images

