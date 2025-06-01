import os
from csv import excel

import torch
import json
from tensorboard.compat.tensorflow_stub.tensor_shape import vector
from torchvision.io import read_image
import pandas as pd
from MachineLearningModels.SDE_conditioned.script_util import create_model
from script_util import create_model_diffusion

from SDE_datareduction import get_test_data, get_data
from SDE_dataclass import LabeledDataset
from main import IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH, BASE_OUTPUT
from SDE_test import forward_process_image
from torch.utils.data import DataLoader, Subset
from SDE_utils import *
from SDE_test import mae, count_parameters

# train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH, **parameters)
#
# _, _, vectors, tensor = next(iter(train_dataloader))
#
# vector_list = [value for value in vectors.values()]
#
# stacked_tensor = torch.stack(vector_list, dim=1)
#
# dataset = LabeledDataset(IMAGE_DATASET_PATH, STRUCTURE_DATASET_PATH)
# converted_image = tesla_to_tensor_jet(image_tensor_b, 8e-3)*255
# image_path = r"C:\Users\tabor\Documents\Programming\MachineLearning\data\figure_B_fixrange\Output\hw1_0.1_hw2_0.11_dww_ii_x_0.002_dww_oo_x_0.002_dww_x_0.05_lcore_x1_IW_0.01_dcs_0.04_hw_0.165_dw_0.195.png"
#
# image_tensor = read_image(image_path)
# image_tensor_b = tensor_jet_to_tesla(image_tensor, 8e-3)

# error = mae(image_tensor.numpy(), converted_image.numpy())

# model, _ = create_model_diffusion("cpu", **config)
# parameters = count_parameters(model)

# with open('config.json', "r", encoding="utf-8") as f:
#     config = json.load(f)
#
# train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(image_dataset_path=IMAGE_DATASET_PATH,
#                                                                       structure_dataset_path=STRUCTURE_DATASET_PATH, **config)
# images = []
# for batch in train_dataloader:
#     image_batch = batch[0]
#     image_batch = tensor_to_PIL(image_batch)
#     images.extend(image_batch)
#
# path = r"results\unsorted_dataset"
# if not os.path.exists(path):
#     os.makedirs(path)
#
# save_image_list(images, path)
#
# print(f"Train Dataloader: {len(train_dataloader.dataset)}")
# print(f"Val Dataloader: {len(val_dataloader.dataset)}")
# print(f"Test Dataloader: {len(test_dataloader.dataset)}")

# Add information to model results dict to save to excel
model_results = {}
model_results['train id'] = 1234
model_results['bm ssim'] = 0.112
model_results['bm mae'] = 0.113
model_results['bm epoch'] = 994

df_model_results = pd.DataFrame([model_results])

excel_results = os.path.join(BASE_OUTPUT, "results.xlsx")

try:
    # Write results to first white row of excel file
    with pd.ExcelWriter(excel_results, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        df_model_results.to_excel(writer, sheet_name='Results', index=False, header=False,
                                  startrow=writer.sheets[
                                      'Results'].max_row if 'Results' in writer.sheets else 0)
except FileNotFoundError:
    # If the file does not yet exist, create a new excel file
    print(f"Error: The file '{excel_results}' was not found. Creating a new file.")
    with pd.ExcelWriter(excel_results, mode='w', engine='openpyxl') as writer:
        df_model_results.to_excel(writer, sheet_name='Results', index=False)