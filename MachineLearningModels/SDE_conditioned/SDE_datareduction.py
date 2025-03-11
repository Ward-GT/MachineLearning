import os
import random
import platform
import numpy as np
import matplotlib.pyplot as plt
from SDE_dataclass import LabeledDataset
from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity as ssim
from SDE_utils import *

def normalize_softmax(x):
    min = x.min(0, keepdims=True)
    max = x.max(0, keepdims=True)
    x = (x - min) / (max - min)
    np.fill_diagonal(x, -np.inf)
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)

def SSIM_matrix(dataset1, dataset2):

    matrix = np.zeros((len(dataset1), len(dataset2)))

    for i, (input_image1, label_image1, _) in enumerate(dataset1):
        print(f"Calculation step: {i}")
        input_image1 = tensor_to_PIL(input_image1)

        for j, (input_image2, label_image2, _) in enumerate(dataset2):
            input_image2 = tensor_to_PIL((input_image2))
            matrix[i,j] = ssim(np.array(input_image1), np.array(input_image2), channel_axis=0, multichannel=True)

    return normalize_softmax(matrix)

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
    total_matrix = np.zeros((len(dimensions_list1), len(dimensions_list2)))

    for i, dimension_dict1 in enumerate(dimensions_list1):
        core_vector1, winding_vector1, total_vector1 = extract_dimension_vectors(dimension_dict1)

        for j, dimension_dict2 in enumerate(dimensions_list2):
            core_vector2, winding_vector2, total_vector2 = extract_dimension_vectors(dimension_dict2)
            core_matrix[i, j] = scaled_dot_product(core_vector1, core_vector2)
            winding_matrix[i, j] = scaled_dot_product(winding_vector1, winding_vector2)
            total_matrix[i, j] = scaled_dot_product(total_vector1, total_vector2)

    multiplied_matrix = core_matrix * winding_matrix
    return normalize_softmax(multiplied_matrix), normalize_softmax(total_matrix)

def calculate_dot_matrix_datasets(dataset1, dataset2):
    dimensions_list1 = [item[2] for item in dataset1]
    dimensions_list2 = [item[2] for item in dataset2]

    multiplied_matrix, total_matrix = calculate_dot_matrix(dimensions_list1, dimensions_list2)

    return multiplied_matrix, total_matrix

def show_similarity_pair(dataset, similarity_vector, indices):

    for index in indices:
        image1, _, _ = dataset[index]
        image2, _, _ = dataset[similarity_vector[index]]

        image1, image2 = np.transpose(tensor_to_PIL(image1), (1, 2, 0)), np.transpose(tensor_to_PIL(image2), (1, 2, 0))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(image1)
        axs[1].imshow(image2)

        plt.show()

def flatten_similarity(similarity_matrix: np.array, starting_index: int = 0):
    similarity_matrix_copy = similarity_matrix.copy()
    index = starting_index

    indices = []
    similarities = []
    for i in range(len(similarity_matrix_copy)):
        similarity = np.max(similarity_matrix_copy[:, index])
        similarities.append(similarity)
        indices.append(index)
        similarity_matrix_copy[index, :] = 0
        index = np.argmax(similarity_matrix_copy[:, index])

    return indices, similarities

def optimize_flatten_similarity(similarity_matrix: np.array, optimization_steps: int):
    similarity_matrix_copy = similarity_matrix.copy()
    starting_indices = random.sample(range(0, len(similarity_matrix_copy)), optimization_steps)
    max_indices = []
    max_similarities = []
    max_similarity = sum(max_similarities)
    print(f"Starting flatten optimization with {optimization_steps} steps")
    for i, starting_index in enumerate(starting_indices):
        indices, similarities = flatten_similarity(similarity_matrix_copy, starting_index)
        similarity = sum(similarities)
        if similarity > max_similarity:
            max_indices = indices
            max_similarities = similarities
            max_similarity = similarity
        print(f"Optimization step: {i}, Similarity: {similarity}, Max Similarity: {max_similarity}")

    return max_indices, max_similarities

def save_ordered_dataset(dataset, indices, path):
    images = []

    for index in indices:
        image, _, _ = dataset[index]
        image = np.transpose(tensor_to_PIL(image), (1, 2, 0))
        image = Image.fromarray(image)
        images.append(image)

    save_image_list(images, path)

def smart_data_split(dataset, train_size: int, val_size: int, test_size: int, optimization_steps: int = 500):
    _, total_matrix = calculate_dot_matrix_datasets(dataset, dataset)

    indices, similarities = optimize_flatten_similarity(total_matrix, optimization_steps)

    evenly_spaced_numbers = np.linspace(0, len(indices) - 1, train_size)

    train_indices = [indices[int(index)] for index in evenly_spaced_numbers]

    all_indices = list(range(len(indices)))

    remaining_indices = [index for index in all_indices if index not in train_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset, test_dataset = torch.utils.data.random_split(Subset(dataset, remaining_indices), [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def get_data(image_dataset_path: str, structure_dataset_path: str, result_path: str = None, split: bool = True, **kwargs):
    image_size = kwargs.get("image_size")
    test_split = kwargs.get("test_split")
    validation_split = kwargs.get("validation_split")
    smart_split = kwargs.get("smart_split")
    batch_size = kwargs.get("batch_size")

    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 2

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = LabeledDataset(image_dataset_path, structure_dataset_path, transform=data_transform)

    if split == True:
        train_size = int((1 - test_split - validation_split) * len(dataset))
        val_size = int(validation_split * len(dataset))
        test_size = int(test_split * len(dataset))

        if smart_split == True:
            train_dataset, val_dataset, test_dataset = smart_data_split(dataset, train_size, val_size, test_size)
        else:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

        if result_path is not None:
            torch.save(train_dataset.indices, os.path.join(result_path, "train_indices.pth"))
            torch.save(val_dataset.indices, os.path.join(result_path, "val_indices.pth"))
            torch.save(test_dataset.indices, os.path.join(result_path, "test_indices.pth"))

        return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        return dataloader, dataset, -1, -1, -1, -1

def get_test_data(test_path: str, image_size: int, batch_size: int, image_dataset_path: str, structure_dataset_path: str):
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 2

    dataset = LabeledDataset(image_dataset_path, structure_dataset_path, transform=data_transform)

    test_indices = torch.load(test_path)
    print(f"Loaded {len(test_indices)} test indices")
    test_subset = Subset(dataset, test_indices)
    print(f"Made subset with {len(test_subset)} images")
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    return test_dataloader