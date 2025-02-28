import torch
import os
import json
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from SDE_train import ModelTrainer
from SDE_datareduction import get_data
from SDE_test import sample_model_output, calculate_metrics, sample_save_metrics
from SDE_utils import save_image_list, set_seed, load_images, tensor_to_PIL
from script_util import create_model_diffusion

DEFAULT_SEED = 42

# Base Paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"Script Dir {SCRIPT_DIR}")
BASE_OUTPUT = r"/home/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results"
# BASE_OUTPUT = "results"

# BASE_INPUT = r"C:\Users\tabor\Documents\Programming\MachineLearning\Data"
BASE_INPUT = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_fixrange")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_maxrange_5000")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE == "cuda" else False

# Test settings
TESTING = False
CALCULATE_METRICS = False
SAMPLE_METRICS = True
TEST_PATH = r"/home/20234635/MachineLearningGit/MachineLearningModels/SDE_conditioned/results/UNet_nblocks_2_noisesteps_1000_smartsplit_False_1/"
SAMPLE_MODEL = "best_model.pth"
NR_SAMPLES = 288

# Training settings
TRAINING = True
SMART_SPLIT = False
GENERATE_IMAGES = True
THRESHOLD_TRAINING = False
LEARN_SIGMA = True
CLIP_GRAD = False
VECTOR_CONDITIONING = False
MIXED_PRECISION = True

# Training parameters
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
EPOCHS = 1000
BATCH_SIZE = 32
IMAGE_SIZE = 128
INIT_LR = 0.00002
WEIGHT_DECAY = 0.001
THRESHOLD = 0.01
EMA_DECAY = 0.9999

# Sampling parameters
NOISE_STEPS = 1000
EMA = False

# UNet Parameters
# MODEL_NAME = "UNet"
# MODEL_NAME = "MiddleUNet"
MODEL_NAME = "SmallUNet"
# MODEL_NAME = "SimpleUNet"
N_BLOCKS = 2
N_HEADS = 4
DIM_HEAD = None
ATTENTION_RESOLUTIONS = "32,16"
N_CHANNELS = 64

parameters = {
    "model_name": MODEL_NAME,
    "smart_split": SMART_SPLIT,
    "threshold_training": THRESHOLD_TRAINING,
    "test_split": TEST_SPLIT,
    "validation_split": VALIDATION_SPLIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "init_lr": INIT_LR,
    "weight_decay": WEIGHT_DECAY,
    "threshold": THRESHOLD,
    "ema_decay": EMA_DECAY,
    "noise_steps": NOISE_STEPS,
    "n_blocks": N_BLOCKS,
    "n_heads": N_HEADS,
    "dim_head": DIM_HEAD,
    "learn_sigma": LEARN_SIGMA,
    "attention_resolutions": ATTENTION_RESOLUTIONS,
    "n_channels": N_CHANNELS,
    "clip_grad": CLIP_GRAD,
    "vector_conditioning": VECTOR_CONDITIONING,
    "mixed_precision": MIXED_PRECISION
}

if TRAINING:
    run_inst = 0
    RUN_NAME = f"{MODEL_NAME}_nblocks_{N_BLOCKS}_noisesteps_{NOISE_STEPS}_smartsplit_{SMART_SPLIT}_{run_inst}"
    while os.path.exists(os.path.join(BASE_OUTPUT, RUN_NAME)):
        run_inst += 1
        RUN_NAME = f"{MODEL_NAME}_nblocks_{N_BLOCKS}_noisesteps_{NOISE_STEPS}_smartsplit_{SMART_SPLIT}_{run_inst}"

    # Output paths
    RESULT_PATH = os.path.join(BASE_OUTPUT, RUN_NAME)
    MODEL_PATH = os.path.join(RESULT_PATH, "models")
    IMAGE_PATH = os.path.join(RESULT_PATH, "images")
    SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
    REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
    STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")
    COLORBAR_PATH = os.path.join(IMAGE_PATH, "Colorbar")

    set_seed(seed=DEFAULT_SEED)
    print(f"Name: {RUN_NAME}")

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    with open(os.path.join(RESULT_PATH, 'parameters.json'), "w") as f:
        json.dump(parameters, f, indent=4)

    model, diffusion = create_model_diffusion(DEVICE, **parameters)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH, result_path=RESULT_PATH, **parameters)
    trainer = ModelTrainer(model=model,
                           device=DEVICE,
                           optimizer=optimizer,
                           image_path=IMAGE_PATH,
                           model_path=MODEL_PATH,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           test_dataloader=test_dataloader,
                           diffusion=diffusion,
                           ema=EMA,
                           **parameters)

    trainer.train()

    torch.save(trainer.best_model_checkpoint, os.path.join(MODEL_PATH, "best_model.pth"))
    if trainer.ema == True and trainer.ema_model is not None:
        torch.save(trainer.ema_model.state_dict(), os.path.join(MODEL_PATH, "ema_model.pth"))

    max_ssim = max(trainer.ssim_values)
    print(f"Max SSIM: {max_ssim}, At place: {5 * np.argmax(trainer.ssim_values)+4}")
    min_mae = min(trainer.mae_values)
    print(f"Min MAE: {min_mae}, At place: {5 * np.argmin(trainer.mae_values)+4}")
    print(f"Best Model Epoch: {trainer.best_model_epoch}")

    np.savez(os.path.join(RESULT_PATH, "train_losses.npz"), losses=trainer.train_losses)
    np.savez(os.path.join(RESULT_PATH, "val_losses.npz"), losses=trainer.val_losses)
    np.savez(os.path.join(RESULT_PATH, "ssim_values.npz"), losses=trainer.ssim_values)
    np.savez(os.path.join(RESULT_PATH, "mae_values.npz"), losses=trainer.mae_values)

    plt.figure(figsize=(12, 6))
    plt.plot(trainer.train_losses[1:], label='Train Loss')
    plt.plot(trainer.val_losses[1:], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(RESULT_PATH, "losses.png"))
    plt.show()

    x_values = range(0, len(trainer.ssim_values) * 5, 5)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, trainer.ssim_values, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.savefig(os.path.join(RESULT_PATH, "SSIM.png"))

    x_values = range(0, len(trainer.mae_values) * 5, 5)
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, trainer.mae_values, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE over Epochs')
    plt.savefig(os.path.join(RESULT_PATH, "MAE.png"))

    if GENERATE_IMAGES == True:
        if EMA == True:
            model = trainer.ema_model
        else:
            model.load_state_dict(trainer.best_model_checkpoint)
        sample_save_metrics(model=model,
                            device=DEVICE,
                            sampler=trainer.diffusion,
                            n=(len(test_dataloader)-1) * BATCH_SIZE,
                            test_dataloader=test_dataloader,
                            result_path=RESULT_PATH, **parameters)

if TESTING:
    PARAMETER_PATH = os.path.join(TEST_PATH, 'parameters.json')
    MODEL_PATH = os.path.join(os.path.join(TEST_PATH, "models"), SAMPLE_MODEL)
    IMAGE_PATH = os.path.join(TEST_PATH, "images")
    SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
    REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
    STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")
    TEST_DATASET_PATH = os.path.join(TEST_PATH, "test_indices.pth")

    if CALCULATE_METRICS == True:
        structure_images = load_images(STRUCTURE_PATH)
        reference_images = load_images(REFERENCE_PATH)
        sampled_images = load_images(SAMPLE_PATH)
        ssim_values, psnr_values, mae_values, max_error_values = calculate_metrics(reference_images[0:1], sampled_images[0:1])
        print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MAE: {np.mean(mae_values)}, Max Error: {np.max(max_error_values)}")

    if SAMPLE_METRICS == True:
        with open(PARAMETER_PATH, "r") as f:
            parameters = json.load(f)

        print(f"Sampling model: {MODEL_PATH}")
        set_seed(seed=DEFAULT_SEED)
        model, sampler = create_model_diffusion(DEVICE, **parameters)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        sample_save_metrics(model=model,
                            device=DEVICE,
                            sampler=sampler,
                            image_dataset_path=IMAGE_DATASET_PATH,
                            structure_dataset_path=STRUCTURE_DATASET_PATH,
                            test_path=TEST_PATH,
                            batch_size_set=BATCH_SIZE,
                            n=NR_SAMPLES,
                            **parameters)