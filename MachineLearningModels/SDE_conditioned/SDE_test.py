from random import sample
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image, ImageColor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from SDE_utils import *
from SDE_datareduction import get_data, get_test_data
from SDE_tools import DiffusionTools

def sample_model_output(
        model: torch.nn.Module,
        device: torch.device,
        sampler: DiffusionTools,
        dataloader: DataLoader,
        n: int,
        batch_size: int,
        vector_conditioning: bool = False
):
    references_list = []
    generated_list = []
    structures_list = []
    iterator = iter(dataloader)
    print(f"Sampling on {device}")

    for i in range(0, n, batch_size):
        if n - i < batch_size:
            print(f"Stopping at index {i} because remaining items ({n - i}) are less than batch_size ({batch_size}).")
            break  # Exit the loop
        references, structures, _, vectors = next(iterator)
        structures = structures.to(device)
        references = references.to(device)
        vectors = vectors.to(device)
        y = (structures if not vector_conditioning else vectors)
        generated, structures = sampler.p_sample_loop(model=model, n=batch_size, y=y)
        references = tensor_to_PIL(references)
        generated = tensor_to_PIL(generated)
        structures = tensor_to_PIL(structures)

        references_list.extend(references)
        generated_list.extend(generated)
        structures_list.extend(structures)
        print(f"Reference: {len(references_list)}, Generated: {len(generated_list)}, Structures: {len(structures_list)}")

    generated_list = [convert_grey_to_white(image) for image in generated_list]

    return references_list, generated_list, structures_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def mse(imageA, imageB):
    # The 'mean' function calculates the average of the array elements.
    # The 'square' function calculates the squared value of each element.
    # np.subtract(imageA, imageB) computes the difference between the images.
    err = np.square(np.subtract(imageA, imageB))
    mean_err = np.mean(err)/255**2
    max_err = np.max(err)/255**2
    return mean_err, max_err

def mae_effective_area(imageA, imageB, threshold=254):
    """
    Calculates the Mean Absolute Error (MAE) between two images,
    considering only the effective area (excluding white parts).

    Args:
        imageA: First image (numpy array).
        imageB: Second image (numpy array).
        threshold: Threshold to determine white pixels (default: 250).

    Returns:
        MAE value on the effective area.
    """

    # Create a mask to identify the effective area (non-white pixels)
    mask = (imageA[:, :, 0] < threshold) & (imageA[:, :, 1] < threshold) & (imageA[:, :, 2] < threshold) | \
           (imageB[:, :, 0] < threshold) & (imageB[:, :, 1] < threshold) & (imageB[:, :, 2] < threshold)

    # Apply the mask to both images (for MAE calculation)
    effective_imageA = imageA[mask]
    effective_imageB = imageB[mask]

    return mae(effective_imageA, effective_imageB)

def mae(imageA, imageB):
    mae = np.mean(np.abs(np.subtract(imageA.astype(np.float32), imageB.astype(np.float32))))/255
    max_error = np.max(np.abs(np.subtract(imageA.astype(np.float32), imageB.astype(np.float32))))/255
    return mae, max_error

def calculate_metrics(image_set1: list[Image.Image], image_set2: list[Image.Image]):

    if len(image_set1) != len(image_set2):
        raise ValueError("Number of images in image sets do not match")

    ssim_values = []
    psnr_values = []
    mae_values = []
    max_error_values = []

    for i in range(len(image_set1)):
        ssim_values.append(ssim(np.array(image_set1[i]), np.array(image_set2[i]), channel_axis=-1, multichannel=True))
        psnr_values.append(psnr(np.array(image_set1[i]), np.array(image_set2[i]), data_range=np.array(image_set1[i]).max() - np.array(image_set1[i]).min()))
        mae, max_error = mae_effective_area(np.array(image_set1[i]), np.array(image_set2[i]))
        mae_values.append(mae)
        max_error_values.append(max_error)

    return ssim_values, psnr_values, mae_values, max_error_values

def sample_save_metrics(
        model: torch.nn.Module,
        device: torch.device,
        sampler: DiffusionTools,
        image_dataset_path = None,
        structure_dataset_path = None,
        test_path: str = None,
        result_path: str = None,
        test_dataloader: DataLoader = None,
        batch_size_set: int = None,
        **kwargs
):

    image_size = kwargs.get("image_size")
    vector_conditioning = kwargs.get("vector_conditioning")
    if batch_size_set == None:
        batch_size = kwargs.get("batch_size")
    else:
        batch_size = batch_size_set

    if test_dataloader is not None and test_path is None:
        print("Using test dataloader")
        image_path = os.path.join(result_path, "images")
        dataloader = test_dataloader
    elif test_path is not None and test_dataloader is None:
        print("Using test data")
        image_path = os.path.join(test_path, "images")
        test_dataset_path = os.path.join(test_path, "test_indices.pth")
        dataloader = get_test_data(test_path=test_dataset_path, image_size=image_size, batch_size=batch_size,
                                   image_dataset_path=image_dataset_path, structure_dataset_path=structure_dataset_path)

    n = (len(dataloader)-1) * batch_size
    sample_path = os.path.join(image_path, "Samples")
    reference_path = os.path.join(image_path, "References")
    structure_path = os.path.join(image_path, "Structures")

    paths = [sample_path, reference_path, structure_path]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    parameter_count = count_parameters(model)

    references, samples, structure = sample_model_output(model=model, device=device, sampler=sampler, n=n, dataloader=dataloader, batch_size=batch_size, vector_conditioning=vector_conditioning)

    ssim_values, psnr_values, mae_values, max_error_values = calculate_metrics(references, samples)

    model_results = {
        'ssim': np.mean(ssim_values),
        'psnr': np.mean(psnr_values),
        'mae': np.mean(mae_values),
        'max_error': np.max(max_error_values)
    }

    print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MAE: {np.mean(mae_values)}, Max Error: {np.max(max_error_values)}, Parameters: {parameter_count}")

    save_image_list(references, reference_path)
    save_image_list(samples, sample_path)
    save_image_list(structure, structure_path)

    return model_results

def create_colorbar_plot(image, max_bfield):

    image_array = np.array(image)

    fig, ax = plt.subplots()

    im = ax.imshow(image_array, cmap=cm.jet)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Magnetic Field Strength (T)')

    ticks_locations = np.linspace(0, 255, num=5)
    tick_labels = np.linspace(0, max_bfield, num=5)
    cbar.set_ticks(ticks_locations, labels=tick_labels)

    return fig

def add_colorbar_to_list(images):
    colorbar_figures = []
    for image in images:
        fig = create_colorbar_plot(image, 8e-3)
        colorbar_figures.append(fig)

    return colorbar_figures

def calculate_error_image(reference: Image, sample: Image):
    reference_array = np.array(reference)
    sample_array = np.array(sample)

    error_array = np.abs(reference_array-sample_array)

    error_image = Image.fromarray(error_array)

    return error_image

def error_image(structure: Image, reference: Image, sample: Image):
    """
    Generates an error image highlighting differences between reference and sample images,
    overlayed onto the structure image. Uses a continuous red-to-green color gradient
    for error magnitudes.

    Args:
        structure: Image representing the underlying structure.
        reference: Reference image for comparison.
        sample: Sample image to compare against the reference.

    Returns:
        Image with the error visualization.
    """

    sample_data = sample.getdata()
    reference_data = reference.getdata()
    structure_data = structure.getdata()
    mask_data = []

    for pixel1, pixel2, pixel3 in zip(sample_data, reference_data, structure_data):
        diffs = [abs(pixel1[i] - pixel2[i]) for i in range(len(pixel1))]
        mae = sum(diffs) / (len(pixel1) * 255) * 100

        # Calculate color based on continuous gradient
        if mae > 6:  # Cap at 5% for red
            mae = 6
        hue = 120 - mae * 20  # 120 is green, 0 is red
        color_hex = ImageColor.getrgb(f"hsl({hue}, 100%, 50%)")  # Full saturation, 50% lightness

        if mae > 0.1:  # Only apply color to errors above 1%
            mask_data.append(color_hex)
        else:
            mask_data.append(pixel3)  # Use structure color for low errors

    mask = Image.new(sample.mode, sample.size)
    mask.putdata(mask_data)
    return mask

def comparison_plot(structures: list, references: list, samples: list, path: str = None, cbar_width=0.02):
    """
    Creates a comparison plot with reference, sample, and error images, including a color bar for the error visualization.
    """
    fig, axs = plt.subplots(len(structures), 3, figsize=(9, 9))

    # Set column titles
    axs[0, 0].set_title('Exact')
    axs[0, 1].set_title('Prediction')
    axs[0, 2].set_title('Error')

    # Create the colormap and normalization for the color bar
    mae_values = np.linspace(0, 6, 256)  # MAE range from 1% to 5%
    colors = []
    for mae in mae_values:
        hue = 120 - (mae) * 20  # 120 is green, 0 is red
        color_rgb = ImageColor.getrgb(f"hsl({hue}, 100%, 50%)")
        color_normalized = tuple(c / 255 for c in color_rgb)
        colors.append(color_normalized)

    cmap = mcolors.LinearSegmentedColormap.from_list('error_cmap', colors)
    norm = mcolors.Normalize(vmin=0, vmax=6)  # Normalize to 1%-5% range

    for i in range(len(structures)):
        # Get the image dimensions
        height, width = references[i].size

        # Plot the reference image
        axs[i, 0].imshow(references[i], extent=[0, width, height, 0])

        # Plot the sample image
        axs[i, 1].imshow(samples[i], extent=[0, width, height, 0])

        # Calculate and plot the error image
        error_img = error_image(structures[i], references[i], samples[i])
        im = axs[i, 2].imshow(error_img, extent=[0, width, height, 0])

    # Add the color bar at the end (after all error images are plotted)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs, orientation='vertical',
                        fraction=cbar_width, pad=0.05, label='Absolute Error (%)')

    if path is not None:
        plt.savefig(path)

    plt.show()
    return

def forward_process_image(sampler, dataloader, device):
    nr_images = 5
    images, _, _ = next(iter(dataloader))
    images = images.to(device)
    images_list = []
    noise_steps = sampler.noise_steps
    images_list.append(tensor_to_PIL(images)[0])

    for t in range(0, noise_steps, noise_steps // nr_images):
        t = sampler.get_specific_timesteps(t, images.shape[0])
        noised_images, _ = sampler.noise_images(images, t)
        noised_images = tensor_to_PIL(noised_images)
        images_list.append(noised_images[0])

        # Calculate the figure size
        fig_width = nr_images * images_list[0].size[0] / 100  # Convert pixels to inches
        fig_height = images_list[0].size[1] / 100  # Convert pixels to inches

    fig, axs = plt.subplots(1, len(images_list))

    for i, img in enumerate(images_list):
        axs[i].imshow(img)
        axs[i].axis('off')  # Hide axes for better visualization

    plt.show()
    return