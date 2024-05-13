import torch
import os
import numpy as np

# Base Paths
BASE_OUTPUT = "results"
# BASE_OUTPUT = r"E:\Ward Taborsky\results"

BASE_INPUT = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data"
# BASE_INPUT = r"E:\Ward Taborsky"
# BASE_INPUT = r"/home/tue/20234635/MachineLearningGit/MachineLearningModels/data"

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined_small")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training parameters
TRAINING = True
SMART_SPLIT = True
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
EPOCHS = 500
NOISE_STEPS = 1000
BATCH_SIZE = 10
IMAGE_SIZE = 128
INIT_LR = 0.00002
WEIGHT_DECAY = 0.001
REFERENCE_IMAGES = True
NR_SAMPLES = 5
GENERATE_IMAGES = False
DEFAULT_SEED = 42

# UNet Parameters
MODEL = "UNet"
N_BLOCKS = 2
TIME_EMB_DIM = 128

RUN_NAME = f"{MODEL}_nblocks_{N_BLOCKS}_smartsplit_{SMART_SPLIT}_split_{TEST_SPLIT}_imgsize_{IMAGE_SIZE}_epochs_{EPOCHS}"
# RUN_NAME = "UNet_ConditionedCombined_1res_01_256_500"

# Output paths
RESULT_PATH = os.path.join(BASE_OUTPUT, RUN_NAME)
MODEL_PATH = os.path.join(RESULT_PATH, "models")
IMAGE_PATH = os.path.join(RESULT_PATH, "images")
SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")

if TRAINING:
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    if not os.path.exists(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)

    if not os.path.exists(REFERENCE_PATH):
        os.makedirs(REFERENCE_PATH)

    if not os.path.exists(STRUCTURE_PATH):
        os.makedirs(STRUCTURE_PATH)
