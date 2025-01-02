import os

import torch
from tensorboard.compat.tensorflow_stub.tensor_shape import vector
from torchvision.io import read_image

from SDE_datareduction import get_test_data, get_data
from SDE_dataclass import LabeledDataset
from config import IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, parameters
from script_util import create_model_diffusion
from SDE_test import forward_process_image
from torch.utils.data import DataLoader, Subset
from SDE_utils import *
from SDE_test import mae

# train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH, **parameters)
#
# _, _, vectors, tensor = next(iter(train_dataloader))
#
# vector_list = [value for value in vectors.values()]
#
# stacked_tensor = torch.stack(vector_list, dim=1)
#
# dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH)

image_path = r"C:\Users\tabor\Documents\Programming\MachineLearning\Data\figure_B_fixrange\Output\hw1_0.1_hw2_0.11_dww_ii_x_0.002_dww_oo_x_0.002_dww_x_0.05_lcore_x1_IW_0.01_dcs_0.04_hw_0.165_dw_0.195.png"
image_tensor = read_image(image_path)
image_tensor_b = tensor_jet_to_tesla(image_tensor, 8e-3)
converted_image = tesla_to_tensor_jet(image_tensor_b, 8e-3)*255

error = mae(image_tensor.numpy(), converted_image.numpy())



