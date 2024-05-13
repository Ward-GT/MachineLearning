import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from SDE_tools import DiffusionTools
from SDE_SimpleUNet import SimpleUNet
from SDE_UNet import UNet
from SDE_utils import *
from SDE_datareduction import get_data, get_test_data

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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

def sample_save_metrics(model, sampler, test_path: str, n: int = 200, batch_size: int = 5):
    parameter_count = count_parameters(model)

    references, samples, structure = sample_model_output(model, sampler, n=n, batch_size=batch_size, test_path=test_path)

    ssim_values, psnr_values, mse_mean_values, mse_max_values = calculate_metrics(references, samples)

    print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MSE Mean: {np.mean(mse_mean_values)}, MSE Max: {np.mean(mse_max_values)}, Parameters: {parameter_count}")

    save_image_list(references, REFERENCE_PATH)
    save_image_list(samples, SAMPLE_PATH)
    save_image_list(structure, STRUCTURE_PATH)

def calculate_error_image(reference: Image, sample: Image):
    reference_array = np.array(reference)
    sample_array = np.array(sample)

    error_array = np.abs(reference_array-sample_array)

    error_image = Image.fromarray(error_array)

    return error_image


def comparison_plot(structure: Image, reference: Image, sample: Image, path: str = None):
    # Create a new figure
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    # Get the image dimensions
    height, width = reference.size

    axs[0].imshow(structure, extent=[0, width, height, 0])
    axs[0].set_title('Structure')

    # Plot the reference image
    axs[1].imshow(reference, extent=[0, width, height, 0])
    axs[1].set_title('Exact')

    # Plot the sample image
    axs[2].imshow(sample, extent=[0, width, height, 0])
    axs[2].set_title('Prediction')

    if path is not None:
        plt.savefig(path)

    # Display the plot
    plt.show()


# model_path = r"E:\Ward Taborsky\results\UNet_ConditionedCombined_1res_01_128_500\models\UNet_ConditionedCombined_2res_01_128_500_final.pth"
# test_path = r"E:\Ward Taborsky\results\UNet_ConditionedCombined_1res_01_128_500\test_indices.pth"
# model = UNet(n_blocks=N_BLOCKS)
# model.load_state_dict(torch.load(model_path))
# sampler = DiffusionTools(img_size=IMAGE_SIZE)
# sample_save_metrics(model, sampler, test_path, n=300, batch_size=5)

structure_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\UNet_nblocks_1_split_0.3_imgsize_128_epochs_500\images\Structures"
reference_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\UNet_nblocks_1_split_0.3_imgsize_128_epochs_500\images\References"
sample_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\UNet_nblocks_1_split_0.3_imgsize_128_epochs_500\images\Samples"
structure_images = load_images(structure_path)
reference_images = load_images(reference_path)
sampled_images = load_images(sample_path)
ssim_values, psnr_values, mse_mean_values, mse_max_values = calculate_metrics(reference_images, sampled_images)
print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MSE Mean: {np.mean(mse_mean_values)}, MSE Max: {np.mean(mse_max_values)}")

# comparison_plot(structure_images[0], reference_images[0], sampled_images[0], r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\SDE_conditioned\results\comparison.eps")
