import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from SDE_tools import DiffusionTools
from SDE_SimpleUNet import SimpleUNet
from SDE_UNet import UNet
from SDE_utils import *

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

model_path = r"C:\Users\20202137\PycharmProjects\MachineLearning\MachineLearningModels\SDE_conditioned\results\SDE_ConditionedwTestSpecific_256_500\models\SDE_ConditionedwTestSpecific_256_500_final.pth"
test_path = r"C:\Users\20202137\PycharmProjects\MachineLearning\MachineLearningModels\SDE_conditioned\results\SDE_ConditionedwTestSpecific_256_500\test_indices.pth"
model = UNet(n_blocks=N_BLOCKS)
model.load_state_dict(torch.load(model_path))
sampler = DiffusionTools(img_size=IMAGE_SIZE)
sample_save_metrics(model, sampler, test_path, n=300, batch_size=5)

reference_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\UNet_ConditionedCombined_1res_01_128_500\images\References"
sample_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\UNet_ConditionedCombined_1res_01_128_500\images\Samples"
reference_images = load_images(reference_path)
sampled_images = load_images(sample_path)
ssim_values, psnr_values, mse_mean_values, mse_max_values = calculate_metrics(reference_images, sampled_images)
print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MSE Mean: {np.mean(mse_mean_values)}, MSE Max: {np.mean(mse_max_values)}")

