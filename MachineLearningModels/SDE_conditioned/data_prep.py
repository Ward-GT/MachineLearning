import os
import torch
import re
import pandas as pd
from torchvision import transforms
from PIL import Image

structure_dir = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B_combined\Structure"
output_dir = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B_combined\Output"
mask_dir = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B_specific\Mask"
def rename_images_to_index(folder_path: str):
    """
    Renames all image files in the specified folder to their index.

    Args:
    folder_path (str): The path to the folder containing the images.
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Sort files to maintain a consistent order
    files.sort()

    # Loop through all files and rename them
    for index, file in enumerate(files):
        # Define the new file name using the index, preserving the original file extension
        new_file_name = f"{index}{os.path.splitext(file)[1]}"

        # Define the full path for the original and new file names
        original_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(original_file_path, new_file_path)
        print(f"Renamed '{file}' to '{new_file_name}'")

def rename_images(folder_path: str):
    files = os.listdir(folder_path)

    for file in files:
        new_file_name = file[2:]

        original_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_file_name)

        os.rename(original_file_path, new_file_path)
        print(f"Renamed {file} to {new_file_name}")

def convert_to_binary_mask(input_dir: str, output_dir: str, threshold: int = 200):
    """
    Converts all images in the input directory to binary mask format
    and saves them to the output directory.

    Parameters:
    - input_dir: Path to the directory containing the images to be converted.
    - output_dir: Path to the directory where the binary masks will be saved.
    - threshold: The value above which a pixel will be set to white, and below
                 which it will be set to black.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define a transform to convert images to binary masks
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: (x < threshold / 255.0).to(torch.float)),  # Apply thresholding
    ])

    # List all files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Process each file
    for filename in image_files:
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)

        # Open the image file
        with Image.open(file_path) as img:
            # Apply the transform to the image
            mask_tensor = transform(img)

            # Convert the tensor to PIL image
            mask_image = transforms.ToPILImage()(mask_tensor)

            # Construct the full output file path
            output_file_path = os.path.join(output_dir, filename)

            # Save the binary mask image
            mask_image.save(output_file_path)
            print(f"Saved binary mask to {output_file_path}")

def output_dict_to_excel(dictionary, output_file):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(dictionary, orient='index')

    # Output the DataFrame to an Excel file
    df.to_excel(output_file)

def extract_dimensions_from_filename(filename):
    # Define the pattern to match dimensions and their numerical values
    pattern = r'(o_hw1|hw2|dww_ii_x|dww_oo_x|dww_x|lcore_x1_IW|dcs|hw|dw)_([0-9.]+)'

    # Use re.findall to find all matches of the pattern
    matches = re.findall(pattern, filename)

    # Convert the matches into a dictionary where keys are dimensions and values are numerical values
    dimensions = {dim: float(value) for dim, value in matches}

    return dimensions

def extract_dimensions(input_dir, file_name):
    # List all files in the input directory
    files = os.listdir(input_dir)
    files = [os.path.splitext(file)[0] for file in files]

    # Initialize an empty dictionary to store the dimensions
    dimensions_dict = {}

    # Process each file
    for i, file in enumerate(files):
        print(file)
        # Extract the dimensions from the file name
        dimensions = extract_dimensions_from_filename(file)

        # Store the dimensions in the dictionary
        dimensions_dict[file] = dimensions

        # Add the index to the dictionary
        dimensions_dict[file]["index"] = i

    output_dir = os.path.join(input_dir, file_name)
    output_dict_to_excel(dimensions_dict, output_dir)

    return dimensions_dict

#TODO: Rename images to index
#TODO: Link index to the dimensions of the images

rename_images(structure_dir)
rename_images(output_dir)
