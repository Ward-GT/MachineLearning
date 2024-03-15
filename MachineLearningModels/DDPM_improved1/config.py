import torch
import os

DATASET_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B"
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")

RUN_NAME = "DDPM_Unconditional_64"

TEST_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training parameters
EPOCHS = 100
BATCH_SIZE = 10
IMAGE_SIZE = 64
INIT_LR = 3e-4

# Output paths
RESULT_PATH = "results"
MODEL_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\saved_models"