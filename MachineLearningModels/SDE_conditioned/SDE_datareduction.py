import random
import platform
import os  # Added for os.path.join

import numpy as np  # Explicitly import numpy
import torch  # Explicitly import torch
from torchvision import transforms  # Explicitly import transforms
from PIL import Image  # Explicitly import Image

from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset

from SDE_dataclass import LabeledDataset
from SDE_utils import tensor_to_PIL, extract_dimension_vectors, save_image_list


def normalize_softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization and then softmax to a numpy array.
    The diagonal of the matrix is set to -infinity before softmax to ensure
    an element is not compared with itself.

    Args:
        x: 1D or 2D np.ndarray of type float. If 2D, operations are column-wise.

    Returns:
        A normalized np.ndarray of the same shape as x.
    """
    # Min-max normalization: scales values to the range [0, 1]
    # keepdims=True ensures that min_val and max_val have the same number of dimensions as x
    # for broadcasting purposes.
    min_val = x.min(axis=0, keepdims=True)
    max_val = x.max(axis=0, keepdims=True)

    # Handle cases where max_val and min_val are the same to avoid division by zero.
    denominator = max_val - min_val
    # Avoid division by zero if a column/array has all same values
    denominator[denominator == 0] = 1  # Or some other strategy, like setting x to 0.5

    x_normalized = (x - min_val) / denominator

    # Set diagonal elements to negative infinity.
    # This is typically done when the matrix represents similarities of items with themselves,
    # and we want to ignore self-similarity in subsequent softmax.
    if x_normalized.ndim == 2 and x_normalized.shape[0] == x_normalized.shape[1]:
        np.fill_diagonal(x_normalized, -np.inf)

    # Softmax activation: converts scores to probabilities.
    # Subtracting np.max(x_normalized, axis=0) improves numerical stability by preventing overflow.
    e_x = np.exp(x_normalized - np.max(x_normalized, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)


def SSIM_matrix(dataset1: DataLoader, dataset2: DataLoader) -> np.ndarray:
    """
    Calculates a matrix of Structural Similarity Index (SSIM) between images from two datasets.
    The resulting matrix is then normalized using the normalize_softmax function.

    Args:
        dataset1: A PyTorch DataLoader or Dataset-like object yielding (image_tensor, ...)
        dataset2: A PyTorch DataLoader or Dataset-like object yielding (image_tensor, ...)

    Returns:
        A 2D np.ndarray of size [len(dataset1), len(dataset2)],
        where each entry (i, j) represents the normalized similarity
        between the i-th image of dataset1 and the j-th image of dataset2.
    """
    # Initialize a matrix to store SSIM scores.
    # Dimensions are number of items in dataset1 vs number of items in dataset2.
    matrix = np.zeros((len(dataset1), len(dataset2)))

    # Iterate through each image in the first dataset.
    for i, (input_image1, *_) in enumerate(dataset1):
        print(f"Calculating SSIM: Image {i + 1}/{len(dataset1)} from dataset1")
        # Convert the tensor image to a PIL Image.
        # tensor_to_PIL is assumed to handle potential batch dimensions and channel orders.
        input_image1_pil = tensor_to_PIL(input_image1)

        # Iterate through each image in the second dataset.
        for j, (input_image2, *_) in enumerate(dataset2):
            input_image2_pil = tensor_to_PIL(input_image2)

            # Calculate SSIM between the two images.
            try:
                # Attempt with channel_axis for newer scikit-image versions
                matrix[i, j] = ssim(np.array(input_image1_pil), np.array(input_image2_pil), channel_axis=0,
                                    data_range=np.array(input_image1_pil).max() - np.array(input_image1_pil).min())
            except TypeError:
                # Fallback for older scikit-image versions that use multichannel
                matrix[i, j] = ssim(np.array(input_image1_pil), np.array(input_image2_pil), multichannel=True,
                                    data_range=np.array(input_image1_pil).max() - np.array(input_image1_pil).min())

    # Normalize the resulting SSIM matrix.
    return normalize_softmax(matrix)


def scaled_dot_product(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes the dot product between two vectors, scaled by the product of their L2 norms.
    This is equivalent to the cosine similarity if vectors are centered (mean subtracted).

    Args:
        vector1: np.ndarray representing the first vector.
        vector2: np.ndarray representing the second vector.

    Returns:
        The scaled dot product (cosine similarity) between vector1 and vector2.
        Returns 0 if either norm is zero to avoid division by zero.
    """
    # Compute the dot product.
    dot_product = np.dot(vector1, vector2)

    # Compute the L2 norm (Euclidean length) of each vector.
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Scale the dot product.
    # If either norm is 0, the scaled dot product is 0 to avoid division by zero.
    if norm1 == 0 or norm2 == 0:
        scaled_dot_product_value = 0.0
    else:
        scaled_dot_product_value = dot_product / (norm1 * norm2)

    return float(scaled_dot_product_value)


def calculate_dot_matrix(dimensions_list1: list, dimensions_list2: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates two 2D matrices representing similarity based on scaled dot products
    of dimension vectors. One matrix considers core and winding dimensions separately
    and multiplies their similarities; the other considers total dimensions.

    Args:
        dimensions_list1: A list of dictionaries, where each dictionary contains
                          dimension vectors (e.g., from `extract_dimension_vectors`).
        dimensions_list2: A list of dictionaries, similar to dimensions_list1.

    Returns:
        A tuple containing two normalized np.ndarrays:
        - multiplied_matrix: Similarity matrix where core and winding vector similarities
                             are calculated separately and then multiplied.
        - total_matrix: Similarity matrix based on the scaled dot product of the
                        concatenated 'total' dimension vectors.
    """
    # Initialize matrices to store scaled dot products.
    num_items1 = len(dimensions_list1)
    num_items2 = len(dimensions_list2)

    core_matrix = np.zeros((num_items1, num_items2))
    winding_matrix = np.zeros((num_items1, num_items2))
    total_matrix = np.zeros((num_items1, num_items2))

    # Iterate through each item in the first list of dimensions.
    for i, dimension_dict1 in enumerate(dimensions_list1):
        # Extract core, winding, and total dimension vectors for the first item.
        core_vector1, winding_vector1, total_vector1 = extract_dimension_vectors(dimension_dict1)

        # Iterate through each item in the second list of dimensions.
        for j, dimension_dict2 in enumerate(dimensions_list2):
            # Extract dimension vectors for the second item.
            core_vector2, winding_vector2, total_vector2 = extract_dimension_vectors(dimension_dict2)

            # Calculate scaled dot product for core, winding, and total vectors.
            core_matrix[i, j] = scaled_dot_product(core_vector1, core_vector2)
            winding_matrix[i, j] = scaled_dot_product(winding_vector1, winding_vector2)
            total_matrix[i, j] = scaled_dot_product(total_vector1, total_vector2)

    # Element-wise multiplication of core and winding similarity matrices.
    # This combines the similarities from two different aspects of the dimensions.
    multiplied_matrix = core_matrix * winding_matrix

    # Normalize both resulting matrices using softmax normalization.
    return normalize_softmax(multiplied_matrix), normalize_softmax(total_matrix)


def calculate_dot_matrix_datasets(dataset1, dataset2) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates dimension similarity matrices for items from two datasets.
    It extracts dimension information (assumed to be the third element of each item)
    from each dataset and then calls `calculate_dot_matrix`.

    Args:
        dataset1: A dataset-like object where each item is a tuple/list,
                  and item[2] contains the dimension dictionary.
        dataset2: Similar to dataset1.

    Returns:
        A tuple containing two normalized np.ndarrays from `calculate_dot_matrix`:
        - multiplied_matrix
        - total_matrix
    """
    # Extract the dimension dictionaries from each item in the datasets.
    # It's assumed that the dimension information is the third element (index 2) of each item.
    dimensions_list1 = [item[2] for item in dataset1]
    dimensions_list2 = [item[2] for item in dataset2]

    # Calculate the dot product matrices based on these extracted dimensions.
    multiplied_matrix, total_matrix = calculate_dot_matrix(dimensions_list1, dimensions_list2)

    return multiplied_matrix, total_matrix


def flatten_similarity(similarity_matrix: np.ndarray, starting_index: int = 0) -> tuple[list[int], list[float]]:
    """
    Flattens a 2D similarity matrix into a 1D sequence of indices and their similarities.
    It uses a greedy approach: starting from `starting_index`, it iteratively selects
    the most similar item (column) to the current item (row `index`), then moves to that
    selected item, marking visited items to avoid cycles in a simple way.

    Args:
        similarity_matrix: A square np.ndarray where matrix[i, j] is the similarity
                           between item i and item j. Higher values mean more similar.
        starting_index: The row index to start the flattening process from.

    Returns:
        A tuple containing:
        - indices: A list of original dataset indices representing the flattened order.
        - similarities: A list of similarity scores between consecutive items in the
                        `indices` list (specifically, the similarity that led to choosing
                        the next item).
    """
    # Create a copy to modify, as we'll be zeroing out rows.
    similarity_matrix_copy = similarity_matrix.copy()
    current_item_index = starting_index  # Renamed for clarity

    ordered_indices = []
    corresponding_similarities = []

    num_items = similarity_matrix_copy.shape[0]

    for _ in range(num_items):
        # Append the current item to the ordered list.
        ordered_indices.append(current_item_index)
        similarities_to_current_item = similarity_matrix_copy[:, current_item_index]

        # The similarity score recorded is the highest similarity *to* the current_item_index
        # from any other item (that hasn't been chosen as a predecessor to current_item_index yet).
        # If current_item_index is the start, this is max(similarity_matrix[:,start_index])
        max_similarity_to_current = np.max(similarities_to_current_item)
        corresponding_similarities.append(float(max_similarity_to_current))

        # Mark the current item (row) as "visited" by zeroing out its row.
        # This prevents picking this `current_item_index` again as a successor.
        similarity_matrix_copy[current_item_index, :] = -np.inf

        # Choose next most similar index and jump
        current_item_index = np.argmax(similarity_matrix_copy[:, current_item_index])

    return ordered_indices, corresponding_similarities


def optimize_flatten_similarity(similarity_matrix: np.ndarray, optimization_steps: int) -> tuple[
    list[int], list[float]]:
    """
    Attempts to find a better flattened sequence from a similarity matrix by running
    `flatten_similarity` multiple times with different random starting indices.
    It returns the sequence that yields the highest sum of similarities.

    Args:
        similarity_matrix: A square np.ndarray of item similarities.
        optimization_steps: The number of different starting indices to try.

    Returns:
        A tuple containing:
        - max_indices: The list of indices for the best flattened sequence found.
        - max_similarities: The list of similarities for the best sequence.
    """
    # Make a copy to ensure the original matrix is not modified by flatten_similarity.
    similarity_matrix_copy = similarity_matrix.copy()
    num_items = len(similarity_matrix_copy)

    # Ensure optimization_steps is not more than the number of items.
    optimization_steps = min(optimization_steps, num_items)

    # Randomly sample starting indices without replacement.
    if num_items == 0:
        return [], []
    starting_indices = random.sample(range(num_items), optimization_steps)

    best_indices_sequence = []
    best_similarities_sequence = []
    # Initialize with negative infinity to ensure any valid sum is greater.
    max_sum_of_similarities = -np.inf

    print(f"Starting flatten optimization with {optimization_steps} steps")
    for i, start_idx in enumerate(starting_indices):
        # Get a flattened sequence and corresponding similarities starting from start_idx.
        current_indices, current_similarities = flatten_similarity(similarity_matrix_copy, start_idx)

        # Calculate the sum of similarities for this sequence.
        current_sum_of_similarities = sum(current_similarities)

        # If this sequence is better than the best found so far, update.
        if current_sum_of_similarities > max_sum_of_similarities:
            best_indices_sequence = current_indices
            best_similarities_sequence = current_similarities
            max_sum_of_similarities = current_sum_of_similarities

        print(f"Optimization step: {i + 1}/{optimization_steps}, Start Index: {start_idx}, "
              f"Current Sum Similarity: {current_sum_of_similarities:.4f}, "
              f"Max Sum Similarity: {max_sum_of_similarities:.4f}")

    return best_indices_sequence, best_similarities_sequence


def save_ordered_dataset(dataset, indices: list[int], path: str):
    """
    Saves images from a dataset in a specified order to a directory.
    The images are saved as a sequence (e.g., for creating a GIF or video).
    Assumes `save_image_list` can handle a list of PIL Images.

    Args:
        dataset: A dataset-like object where each item is (image_tensor, label, dimensions).
        indices: A list of integers representing the desired order of items from the dataset.
        path: The file path (e.g., directory or filename for a composite image)
              where the ordered images should be saved.
    """
    ordered_images_pil = []

    # Iterate through the dataset using the provided indices.
    for index in indices:
        # Retrieve the image tensor from the dataset at the given index.
        # Item structure is assumed to be (image, possibly_label, possibly_metadata).
        image_tensor, _, _ = dataset[index]

        # Convert the tensor to a PIL Image.
        pil_image_or_numpy = tensor_to_PIL(image_tensor)

        if isinstance(pil_image_or_numpy, np.ndarray):
            # If it's a numpy array, ensure it's in H, W, C format for Image.fromarray
            if pil_image_or_numpy.shape[0] == 3 or pil_image_or_numpy.shape[0] == 1:  # Assuming C, H, W
                pil_image_or_numpy = np.transpose(pil_image_or_numpy, (1, 2, 0))
            # Ensure it's in uint8 format if not already
            if pil_image_or_numpy.dtype != np.uint8:
                pil_image_or_numpy = (pil_image_or_numpy * 255).astype(
                    np.uint8) if pil_image_or_numpy.max() <= 1.0 else pil_image_or_numpy.astype(np.uint8)

            final_image = Image.fromarray(pil_image_or_numpy)
        else:  # Assuming it's already a PIL Image
            final_image = pil_image_or_numpy

        ordered_images_pil.append(final_image)

    # Save the list of PIL images.
    # `save_image_list` is a custom utility function.
    save_image_list(ordered_images_pil, path)
    print(f"Saved {len(ordered_images_pil)} images in order to {path}")


def smart_data_split(dataset, train_target_size: int, test_target_size: int) -> tuple[Subset, Subset]:
    """
    Splits a dataset into training and testing subsets using a "smart" approach.
    This involves:
    1. Calculating a similarity matrix based on item dimensions.
    2. Finding an optimized flattened order of items based on this similarity.
    3. Selecting training samples by taking evenly spaced items from this ordered list.
    4. Assigning the remaining items to the test set.

    Args:
        dataset: The full dataset to be split. Items are expected to have dimension
                 information accessible for `calculate_dot_matrix_datasets`.
        train_target_size: The desired number of items in the training set.
        test_target_size: The desired number of items in the test set.
                         Note: The actual test size will be len(dataset) - train_actual_size.

    Returns:
        A tuple containing:
        - train_dataset: A PyTorch Subset for training.
        - test_dataset: A PyTorch Subset for testing.
    """
    print("Starting smart data split...")
    # Calculate the 'total_matrix' which represents similarity based on all dimensions.
    # The 'multiplied_matrix' is calculated but not used in this function.
    _, total_similarity_matrix = calculate_dot_matrix_datasets(dataset, dataset)

    # Optimize the flattening of the similarity matrix to get an ordered sequence of indices.
    # The number of optimization steps is set to the length of the dataset.
    num_items = len(dataset)
    if num_items == 0:
        raise ValueError("Dataset cannot be empty for smart_data_split.")

    # `optimize_flatten_similarity` returns the best sequence of indices and their similarities.
    # We only need the `ordered_indices` here.
    ordered_indices, _ = optimize_flatten_similarity(total_similarity_matrix, optimization_steps=num_items)

    # Select indices for the training set by taking evenly spaced items from the `ordered_indices`.
    # `np.linspace` creates `train_target_size` points distributed from 0 to `len(ordered_indices) - 1`.
    if train_target_size > len(ordered_indices):
        raise ValueError(f"Warning: train_target_size ({train_target_size}) is greater than the number of available ordered items ({len(ordered_indices)})")
    if train_target_size == 0:
        train_sample_indices_in_ordered_list = []
    else:
        train_sample_indices_in_ordered_list = np.linspace(0, len(ordered_indices) - 1, num=train_target_size,
                                                           dtype=int)

    # Map these sample indices back to the original dataset indices from `ordered_indices`.
    train_indices_final = [ordered_indices[int(i)] for i in train_sample_indices_in_ordered_list]

    train_indices_set = set(train_indices_final)

    # Create PyTorch Subset objects for train and test datasets.
    train_dataset = Subset(dataset, train_indices_final)

    # The test set should be the remaining items from the original dataset.
    original_dataset_indices = list(range(num_items))
    test_indices_final = [idx for idx in original_dataset_indices if idx not in train_indices_set]

    # If a specific test_size is required and it's smaller than remaining:
    if len(test_indices_final) > test_target_size and test_target_size > 0:
        raise ValueError(f"Actual remaining items for test: {len(test_indices_final)}. Requested test_target_size: {test_target_size}")

    test_dataset = Subset(dataset, test_indices_final)

    print(f"Smart split: Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, test_dataset


def check_platform_num_workers() -> int:
    """
    Determines the number of worker processes for PyTorch DataLoaders based on the OS.
    Windows often has issues with `num_workers > 0` in certain environments/scripts.

    Returns:
        Number of workers (0 for Windows, 2 for other OS by default).
    """
    if platform.system() == "Windows":
        num_workers = 0
    else:
        # Default to 2 for non-Windows systems, can be tuned.
        num_workers = 2
    print(f"Operating System: {platform.system()}. Number of DataLoader workers set to: {num_workers}")
    return num_workers


def get_image_transform(image_size: int):
    """
    Creates a standard torchvision transform pipeline for images.
    Resizes, normalizes to [0, 1], then to [-1, 1].

    Args:
        image_size: The target size (height and width) for the images.

    Returns:
        A torchvision.transforms.Compose object.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda t: t / 255.0),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])


def get_data(
        image_dataset_path: str,
        structure_dataset_path: str,
        result_path: str = None,
        split: bool = True,
        **kwargs
):
    """
    Loads, transforms, and splits data into training, validation, and test DataLoaders.

    Args:
        image_dataset_path: Path to the image dataset.
        structure_dataset_path: Path to the structural data associated with images.
        result_path: Optional path to save the indices of the splits.
        split: Boolean, if True, split into train/val/test. Otherwise, return a single DataLoader for the whole dataset.
        **kwargs: Keyword arguments including:
            image_size (int): Target image size.
            test_split (float): Proportion of data for the test set (e.g., 0.1 for 10%).
            validation_split (float): Proportion of data for the validation set (e.g., 0.1 for 10%).
            smart_split (bool): If True, use `smart_data_split` for train/test split after validation split.
            batch_size (int): Batch size for DataLoaders.

    Returns:
        If split is True:
            train_dataloader, val_dataloader, test_dataloader,
            train_dataset_subset, val_dataset_subset, test_dataset_subset
        If split is False:
            dataloader, dataset, -1, -1, -1, -1 (placeholders for subset info)
    """
    # Extract parameters from kwargs or set defaults.
    image_size = kwargs.get("image_size", 64)  # Default image size if not provided
    batch_size = kwargs.get("batch_size", 32)  # Default batch size

    # Determine the number of workers for DataLoader.
    num_workers = check_platform_num_workers()

    # Define image transformations.
    data_transform = get_image_transform(image_size)

    # Load the custom LabeledDataset.
    print(f"Loading dataset from: Images='{image_dataset_path}', Structures='{structure_dataset_path}'")
    dataset = LabeledDataset(image_dataset_path, structure_dataset_path, transform=data_transform)
    dataset_size = len(dataset)
    print(f"Total dataset size: {dataset_size} items.")

    if dataset_size == 0:
        raise ValueError("Loaded dataset is empty. Please check paths and data.")

    if split:
        # Extract split ratios from kwargs.
        test_split_ratio = kwargs.get("test_split", 0.1)
        validation_split_ratio = kwargs.get("validation_split", 0.1)
        smart_split_flag = kwargs.get("smart_split", False)

        if not (0 <= test_split_ratio < 1 and 0 <= validation_split_ratio < 1):
            raise ValueError("Split ratios must be between 0 and 1.")
        if test_split_ratio + validation_split_ratio >= 1:
            raise ValueError("Sum of test_split and validation_split ratios must be less than 1.")

        # Calculate sizes for train, validation, and test sets.
        val_size = int(validation_split_ratio * dataset_size)
        test_size = int(test_split_ratio * dataset_size)  # This is the target test size

        # The remaining data after validation split will be further split into train and test.
        remaining_size_after_val = dataset_size - val_size

        # Create a consistent validation dataloader first.
        # A fixed generator seed ensures the validation set is the same across runs.
        generator = torch.Generator().manual_seed(42)  # Use a common seed

        if val_size == 0 and remaining_size_after_val == dataset_size:  # No validation set
            val_dataset = Subset(dataset, [])  # Empty subset
            remaining_dataset = dataset  # All data is remaining
        elif val_size > 0 and remaining_size_after_val > 0:
            remaining_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [remaining_size_after_val, val_size], generator=generator
            )
        else:  # Should not happen if val_size >= 0 and remaining_size_after_val >=0
            raise ValueError("Logic error in validation split sizing.")

        assert len(val_dataset) == val_size, f"Validation set size mismatch: expected {val_size}, got {len(val_dataset)}"

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                    num_workers=num_workers)

        print(f"Validation set size: {len(val_dataset)}")

        # Now, split the `remaining_dataset` into training and testing sets.
        # The `test_size` calculated earlier is the target for the final test set.
        # The `train_size` should be `remaining_size_after_val - test_size`.
        train_size_for_split = remaining_size_after_val - test_size

        if train_size_for_split < 0:  # test_size is too large for what's remaining
            raise ValueError("Train size for split < 0, adjust sizes")

        if smart_split_flag:
            print("Using smart split for train/test...")
            # smart_data_split needs target sizes for train and test from the `remaining_dataset`
            train_dataset, test_dataset = smart_data_split(remaining_dataset, train_size_for_split, test_size)
        else:
            print("Using random split for train/test...")
            # Ensure split sizes are non-negative for random_split
            current_train_size = max(0, train_size_for_split)
            current_test_size = max(0, len(remaining_dataset) - current_train_size)  # Test gets the true remainder

            if current_train_size + current_test_size > len(remaining_dataset):
                # This can happen due to rounding if test_size was calculated from original dataset_size
                # Adjust test_size to be exactly what's left.
                current_test_size = len(remaining_dataset) - current_train_size

            train_dataset, test_dataset = torch.utils.data.random_split(
                remaining_dataset, [current_train_size, current_test_size],
                generator=generator)  # Use same seed for reproducibility

        # Assert actual sizes match targets as closely as possible (considering integer arithmetic)
        # The key is that sum of train, val, test subsets should be original dataset size.
        print(f"Train set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        actual_total_split_size = len(train_dataset) + len(val_dataset) + len(test_dataset)
        if actual_total_split_size != dataset_size:
            print(
                f"Warning: Sum of split dataset sizes ({actual_total_split_size}) does not match original dataset size ({dataset_size}). This might be due to rounding or empty splits.")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=num_workers)

        # Save indices of the subsets if a result path is provided.
        if result_path:
            torch.save(train_dataset.indices, os.path.join(result_path, "train_indices.pth"))
            torch.save(val_dataset.indices, os.path.join(result_path, "val_indices.pth"))
            torch.save(test_dataset.indices, os.path.join(result_path, "test_indices.pth"))
            print(f"Saved split indices to {result_path}")

        return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset

    else:
        # If not splitting, return a DataLoader for the entire dataset.
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        # Return placeholders for subset information.
        return dataloader, dataset, None, None, None, None  # Changed -1 to None for subsets


def get_test_data(
        test_indices_path: str,
        image_size: int,
        batch_size: int,
        image_dataset_path: str,  # Added, as LabeledDataset needs it
        structure_dataset_path: str  # Added, as LabeledDataset needs it
) -> DataLoader:
    """
    Loads a test dataset based on a file containing test indices.

    Args:
        test_indices_path: Path to the .pth file containing a list/tensor of test indices.
        image_size: Target image size for transformations.
        batch_size: Batch size for the DataLoader.
        image_dataset_path: Path to the main image dataset (needed to construct full dataset).
        structure_dataset_path: Path to the main structure dataset (needed to construct full dataset).


    Returns:
        A PyTorch DataLoader for the test set.
    """
    # Define image transformations.
    data_transform = get_image_transform(image_size)

    # Determine number of workers.
    num_workers = check_platform_num_workers()

    # Load the full dataset first.
    print(
        f"Loading full dataset to extract test subset: Images='{image_dataset_path}', Structures='{structure_dataset_path}'")
    full_dataset = LabeledDataset(image_dataset_path, structure_dataset_path, transform=data_transform)

    # Load the pre-saved test indices.
    print(f"Loading test indices from: {test_indices_path}")
    test_indices = torch.load(test_indices_path)
    print(f"Loaded {len(test_indices)} test indices.")

    # Create a Subset using these indices.
    test_subset = Subset(full_dataset, test_indices)
    print(f"Created test subset with {len(test_subset)} images.")

    if len(test_subset) == 0:
        print("Warning: Test subset is empty after loading indices.")

    # Create a DataLoader for the test subset.
    # Shuffle is typically False for test sets to ensure consistent evaluation.
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    return test_dataloader