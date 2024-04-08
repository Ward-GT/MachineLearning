import numpy as np
from SDE_tools import DiffusionTools
from SDE_dataclass import LabeledDataset
from SDE_model import UNet
from SDE_utils import *

model = UNet()
model.load_state_dict(torch.load(r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\SDE_ConditionedwBigTest_128_500\models\SDE_ConditionedwBigTest_128_500_final.pth"))
model.eval()

sampler = DiffusionTools()
references, generated, structure = sample_model_output(model, sampler,n=200, batch_size=5, test_path=r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\SDE_ConditionedwBigTest_128_500\test_indices.pth")

ssim, psnr, mse_mean, mse_max = calculate_metrics(references, generated)
print(np.mean(ssim), np.mean(psnr), np.mean(mse_mean), np.mean(mse_max))

sample_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\SDE_ConditionedwBigTest_128_500\images\Samples"
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
reference_path = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\results\SDE_ConditionedwBigTest_128_500\images\References"
if not os.path.exists(reference_path):
    os.makedirs(reference_path)

save_image_list(references, reference_path)
save_image_list(generated, sample_path)

print(count_parameters(model))