import torch
import os
import numpy as np

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
DEFAULT_SEED = 42

# UNet Parameters
MODEL = "UNet"
N_BLOCKS = 1

# RUN_NAME = f"{MODEL}_nblocks_{N_BLOCKS}_split_{TEST_SPLIT}_imgsize_{IMAGE_SIZE}_epochs_{EPOCHS}"
RUN_NAME = "UNet_ConditionedCombined_2res_01_256_500"

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

def set_seed(seed: int = DEFAULT_SEED, fully_deterministic: bool = False):
    """
    Set seed for reproducible behavior.

    Parameters
    ----------
    seed : int
        Seed value to set. By default, 1958.
    fully_deterministic : bool
        Whether to set the environment to fully deterministic. By default, False.
        This should only be used for debugging and testing, as it can significantly
        slow down training at little to no benefit.
    """
    if fully_deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed set!")

set_seed()
