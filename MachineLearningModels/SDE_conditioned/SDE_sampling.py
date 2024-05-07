import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from SDE_utils import *

def SSIM_matrix(dataset1, dataset2):

    matrix = np.zeros((len(dataset1), len(dataset2)))

    for i, (input_image1, label_image1, _) in enumerate(dataset1):
        input_image1 = tensor_to_PIL(input_image1)
        print(np.array(input_image1).shape)

        for j, (input_image2, label_image2, _) in enumerate(dataset2):
            input_image2 = tensor_to_PIL((input_image2))
            print(np.array(input_image2).shape)
            matrix[i,j] = ssim(np.array(input_image1), np.array(input_image2), channel_axis=0, multichannel=True)

    return matrix

def extract_dimension_vectors(dimension_dict : dict):
    core_keys = ["hw", "dw", 'dcs']
    winding_keys = ["hw1", "hw2", "dww_ii_x", "dw_oo_x", "dww_x", "lcore_x1_IW"]

    core_vector = np.array([dimension_dict[key] for key in core_keys if key in dimension_dict])
    winding_vector = np.array([dimension_dict[key] for key in winding_keys if key in dimension_dict])

    return core_vector, winding_vector


def scaled_dot_product(vector1, vector2):
    # Compute the dot product
    dot_product = np.dot(vector1, vector2)

    # Compute the norms of the vectors
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Scale the dot product
    scaled_dot_product = dot_product / (norm1 * norm2)

    return scaled_dot_product
def calculate_dot_matrix(dimensions_list1: list, dimensions_list2: list):
    core_matrix = np.zeros((len(dimensions_list1), len(dimensions_list2)))
    winding_matrix = np.zeros((len(dimensions_list1), len(dimensions_list2)))

    for i, dimension_dict1 in enumerate(dimensions_list1):
        core_vector1, winding_vector1 = extract_dimension_vectors(dimension_dict1)

        for j, dimension_dict2 in enumerate(dimensions_list2):
            core_vector2, winding_vector2 = extract_dimension_vectors(dimension_dict2)
            core_matrix[i, j] = scaled_dot_product(core_vector1, core_vector2)
            winding_matrix[i, j] = scaled_dot_product(winding_vector1, winding_vector2)

    return core_matrix, winding_matrix

def calculate_dot_matrix_datasets(dataset1, dataset2):
    dimensions_list1 = [item[2] for item in dataset1]
    dimensions_list2 = [item[2] for item in dataset2]

    core_matrix, winding_matrix = calculate_dot_matrix(dimensions_list1, dimensions_list2)

    return core_matrix, winding_matrix

_, dataset, _, _ = get_data(split=False)

matrix = SSIM_matrix(dataset, dataset)

print(matrix)

# _, _, dimensions = dataset[1]
#
# print(dimensions)
# print(extract_dimension_vectors(dimensions))

core_matrix, winding_matrix = calculate_dot_matrix_datasets(dataset, dataset)