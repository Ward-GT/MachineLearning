import os
from SDE_datareduction import get_test_data
from config import IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, parameters
from script_util import create_model_diffusion
from SDE_test import forward_process_image
from torch.utils.data import DataLoader, Subset

result_path = r"C:\Users\tabor\Documents\Programming\MachineLearning\MachineLearningModels\SDE_conditioned\results\ConditionedPriorTesting"
test_path = os.path.join(result_path, "test_indices.pth")
device = "cpu"

dataloader = get_test_data(test_path=test_path, image_size=parameters.get('image_size'), batch_size=10,
                           image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH)

# Define the number of images you want in the new dataloader
num_images = 10

dataset = dataloader.dataset

# Create a subset of the dataset
subset_indices = list(range(num_images))
subset = Subset(dataset, subset_indices)

# Create a new dataloader with the subset
new_dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers)

_, diffusion = create_model_diffusion(device, **parameters)

forward_process_image(diffusion, new_dataloader, device)
