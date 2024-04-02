import torch
import os

RUN_NAME = "SDE_ConditionedwTest_128_500"

# DATASET_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B_specific"
DATASET_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data\figure_B"
# DATASET_PATH = r"E:\Ward Taborsky\figure_B"

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

TEST_SPLIT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training parameters
EPOCHS = 500
NOISE_STEPS = 1000
BATCH_SIZE = 10
IMAGE_SIZE = 128
TIME_DIM = 128
INIT_LR = 0.0001

# Output paths
BASE_OUTPUT = "results"
RESULT_PATH = os.path.join(BASE_OUTPUT, RUN_NAME)
MODEL_PATH = os.path.join(RESULT_PATH, "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
IMAGE_PATH = os.path.join(RESULT_PATH, "images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
LOG_PATH = os.path.join(RESULT_PATH, f"{RUN_NAME},losses.npz")
