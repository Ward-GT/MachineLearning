import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from SDE_tools import DiffusionTools
from SDE_model import UNet
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

model_path = r"C:\Users\20202137\PycharmProjects\MachineLearning\MachineLearningModels\SDE_conditioned\results\SDE_ConditionedwTestSpecific_256_500\models\SDE_ConditionedwTestSpecific_256_500_final.pth"
test_path = r"C:\Users\20202137\PycharmProjects\MachineLearning\MachineLearningModels\SDE_conditioned\results\SDE_ConditionedwTestSpecific_256_500\test_indices.pth"
model = UNet()
model.load_state_dict(torch.load(model_path))

sampler = DiffusionTools(img_size=IMAGE_SIZE)
references, generated, structure = sample_model_output(model, sampler,n=200, batch_size=5, test_path=test_path)

ssim, psnr, mse_mean, mse_max = calculate_metrics(references, generated)

save_image_list(references, REFERENCE_PATH)
save_image_list(generated, SAMPLE_PATH)
save_image_list(structure, STRUCTURE_PATH)

parameter_count = count_parameters(model)