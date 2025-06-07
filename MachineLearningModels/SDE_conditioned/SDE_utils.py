import os
import torch
import pandas as pd
import torchvision
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import cm

def load_images(folder_path: str):
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group())):
        print(filename)
        img = Image.open(os.path.join(folder_path, filename))
        images.append(img)
    return images

def save_image_list(image_list: list, path: str):
    if isinstance(image_list[0], Image.Image):
        for i, image in enumerate(image_list):
            image.save(os.path.join(path, f"{i}.png"))
    elif isinstance(image_list[0], matplotlib.figure.Figure):
        for i, fig in enumerate(image_list):
            fig.savefig(os.path.join(path, f"{i}.png"))
    else:
        raise ValueError("Image type not known")

def tensor_jet_to_tesla(image_tensor, max_b_field):
    """
    Convert jet colormap tensor to Tesla values, handling white pixels

    Parameters:
    -----------
    image_tensor : torch.Tensor
        Input tensor of shape [C, H, W] or [B, C, H, W]
    max_b_field : float
        Maximum magnetic field value

    Returns:
    --------
    torch.Tensor
        Tensor of Tesla values
    """
    # Ensure tensor is in the right format
    if image_tensor.max() > 1:
        image_tensor = image_tensor / 255.0

    # Create jet colormap lookup table
    jet_lut = torch.tensor(cm.jet(np.linspace(0, 1, 256))[:, :3], dtype=image_tensor.dtype, device=image_tensor.device)

    # Reshape
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.permute(1, 2, 0)

    # Flatten the spatial dimensions
    image_flat = image_tensor.reshape(-1, 3)

    # Detect white pixels (RGB values all close to 1)
    white_mask = torch.all(image_flat > 0.95, dim=1)

    # Calculate squared differences for non-white pixels
    diff = torch.sum((image_flat.unsqueeze(1) - jet_lut) ** 2, dim=2)
    indices = torch.argmin(diff, dim=1)

    # Convert indices to intensities and then to Tesla values
    intensities = indices.float() / 255.0
    tesla_values = intensities * max_b_field

    # Set white pixels to 0 Tesla
    tesla_values[white_mask] = 0.0

    # Reshape back to original spatial dimensions
    return tesla_values.reshape(image_tensor.shape[:2]).unsqueeze(-3)


def tesla_to_tensor_jet(tesla_values, max_b_field):
    """
    Convert Tesla values back to jet colormap RGB tensor

    Parameters:
    -----------
    tesla_values : torch.Tensor
        Input tensor of Tesla values
    max_b_field : float
        Maximum magnetic field value

    Returns:
    --------
    torch.Tensor
        RGB tensor with jet colormap values
    """
    # Normalize tesla values to [0, 1]
    intensities = tesla_values / max_b_field
    # Create jet colormap lookup table
    jet_lut = torch.tensor(cm.jet(np.linspace(0, 1, 256))[:, :3], dtype=intensities.dtype, device=intensities.device)

    # Convert intensities to indices
    indices = (intensities * 255).long().clamp(0, 255)

    # Look up RGB values
    rgb_values = jet_lut[indices]

    # Handle zeros (potentially from white pixels) - set to white
    white_mask = tesla_values == 0
    rgb_values[white_mask] = 1.0
    rgb_values = rgb_values.squeeze(-4)

    return rgb_values.moveaxis(-1, -3)

def extract_dimension_vectors(dimension_dict: dict):
    core_keys = ["hw", "dw", 'dcs']
    winding_keys = ["hw1", "hw2", "dww_ii_x", "dw_oo_x", "dww_x", "lcore_x1_IW"]

    core_vector = np.array([dimension_dict[key] for key in core_keys if key in dimension_dict])
    winding_vector = np.array([dimension_dict[key] for key in winding_keys if key in dimension_dict])
    total_vector = np.concatenate((core_vector, winding_vector))

    return core_vector, winding_vector, total_vector

def dimension_vectors_to_tensor(dimension_dict: dict):
    vector_list = [value for value in dimension_dict.values()]
    stacked_tensor = torch.stack(vector_list, dim=1).float()
    return stacked_tensor

def save_images(reference_images: list[Image]=None, generated_images: list[Image]=None, structure_images: list[Image]=None, path: str=None, **kwargs):
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

def concatenate_images(images: torch.Tensor, structures: torch.Tensor):
    return torch.cat((images, structures), dim=1)

def split_images(concatenated_images: torch.Tensor):
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
        image_tensor = tensor[i]
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)  # Remove the batch dimension
        image = transforms.ToPILImage()(image_tensor)
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

def convert_black_to_white(image: Image):
    image_array = np.array(image)

    # Define a function to apply to each pixel
    def change_color(pixel):
        # If all the channels of the pixel value are above the threshold, change it to white
        if np.all(pixel < 5):
            return (255, 255, 255)
        else:
            return pixel

    # Apply the function to each pixel in the image
    new_image_array = np.apply_along_axis(change_color, axis=-1, arr=image_array)
    new_image = Image.fromarray(new_image_array.astype('uint8'), 'RGB')
    return new_image

def set_seed(seed: int, fully_deterministic: bool = False):
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

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def save_json_file(data, filename, indent=4):
    """
    Saves a Python dictionary or list to a JSON file.

    Args:
        data (dict or list): The Python object to be serialized to JSON.
        filename (str): The name of the file to save (e.g., "my_data.json").
        indent (int, optional): The number of spaces to use for indentation
                                in the JSON file. Use None for no indentation
                                (compact output). Defaults to 4.
    """
    try:
        # Ensure the directory exists if the filename includes a path
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"Data successfully saved to '{filename}'")
    except TypeError as e:
        print(f"Error: Cannot serialize data to JSON. Check data types. {e}")
    except IOError as e:
        print(f"Error saving file '{filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

