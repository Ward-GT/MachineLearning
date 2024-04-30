import torch
import os

# Base Paths
BASE_OUTPUT = "results"
# BASE_INPUT = r"C:\Users\20202137\OneDrive - TU Eindhoven\Programming\Python\MachineLearning\MachineLearningModels\data"
# BASE_INPUT = r"E:\Ward Taborsky"
BASE_INPUT = r"/home/tue/20234635/MachineLearningGit/MachineLearningModels/data"

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training parameters
TEST_SPLIT = 0.1
EPOCHS = 500
NOISE_STEPS = 1000
BATCH_SIZE = 20
IMAGE_SIZE = 256
TIME_EMB_DIM = 128
INIT_LR = 0.00002
WEIGHT_DECAY = 0.001
REFERENCE_IMAGES = True
NR_SAMPLES = 5
GENERATE_IMAGES = False

# UNet Parameters
MODEL = "UNet"
N_BLOCKS = 1

# RUN_NAME = f"{MODEL}_nblocks_{N_BLOCKS}_split_{TEST_SPLIT}_imgsize_{IMAGE_SIZE}_epochs_{EPOCHS}"
RUN_NAME = "/home/tue/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results/UNet_ConditionedCombined_2res_01_256_500"

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
