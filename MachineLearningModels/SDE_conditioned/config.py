import torch
import os

# Base Paths
RUN_NAME = "SDE_ConditionedwTestSpecific_256_500"
BASE_OUTPUT = "results"
# BASE_INPUT = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data"
BASE_INPUT = r"E:\Ward Taborsky"

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training parameters
TEST_SPLIT = 0.1
EPOCHS = 200
NOISE_STEPS = 1000
BATCH_SIZE = 10
IMAGE_SIZE = 256
TIME_DIM = 128
INIT_LR = 0.0001
WEIGHT_DECAY = 0.001

# Output paths
RESULT_PATH = os.path.join(BASE_OUTPUT, RUN_NAME)
LOG_PATH = os.path.join(RESULT_PATH, f"{RUN_NAME},losses.npz")
MODEL_PATH = os.path.join(RESULT_PATH, "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

IMAGE_PATH = os.path.join(RESULT_PATH, "images")
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)

REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
if not os.path.exists(REFERENCE_PATH):
    os.makedirs(REFERENCE_PATH)

STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")
if not os.path.exists(STRUCTURE_PATH):
    os.makedirs(STRUCTURE_PATH)
