# import packages
import torch
import os

RUN_NAME = "BinarySegmentation_test"
# paths dataset
DATASET_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B"
# DATASET_PATH = r"E:\Ward Taborsky\figure_B"
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "Mask")

TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# U-net parameters
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3

# Training parameters
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 10

# image dimension
IMAGE_SIZE = 128

# threshold for weak predictions
THRESHOLD = 0.5

BASE_OUTPUT = os.path.join("results", RUN_NAME)
MODEL_PATH = os.path.join(BASE_OUTPUT, f"{RUN_NAME}_model.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, f"{RUN_NAME}_lossplot.png")
LOG_PATH = os.path.join(BASE_OUTPUT, f"{RUN_NAME}_log.npz")
TEST_PATH = os.path.join(BASE_OUTPUT, "test_paths.txt")
