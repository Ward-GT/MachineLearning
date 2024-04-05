import os
import torch
from torchvision import transforms
from PIL import Image

structure_dir = r"/MachineLearningModels/data/figure_B_specific/Structure"
output_dir = r"/MachineLearningModels/data/figure_B_specific/Output"
mask_dir = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B_specific\Mask"
def rename_images_to_index(folder_path):
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

def convert_to_binary_mask(input_dir, output_dir, threshold=200):
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

rename_images_to_index(structure_dir)
rename_images_to_index(output_dir)
# convert_to_binary_mask(structure_dir, mask_dir)