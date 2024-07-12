import torch
import os
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from SDE_train import ModelTrainer
from SDE_UNet import UNet
from SDE_datareduction import get_data
from SDE_tools import DiffusionTools
from SDE_test import sample_model_output
from SDE_utils import save_image_list

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

# Settings
TRAINING = False
SMART_SPLIT = True
REFERENCE_IMAGES = True
GENERATE_IMAGES = True

# Training parameters
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
EPOCHS = 500
BATCH_SIZE = 20
IMAGE_SIZE = 128
INIT_LR = 0.00002
WEIGHT_DECAY = 0.001
DEFAULT_SEED = 42

# Sampling parameters
NOISE_STEPS = 1000
NR_SAMPLES = 5

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
    print(f"Name: {RUN_NAME}, Smart Split : {SMART_SPLIT}")
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

    model = UNet(n_blocks=N_BLOCKS)
    optimizer = optim.AdamW(model.parameters(), lr = INIT_LR, weight_decay=WEIGHT_DECAY)
    sampler = DiffusionTools(noise_steps=NOISE_STEPS, img_size = IMAGE_SIZE, device = DEVICE)
    train_dataloader, val_dataloader, test_dataloader = get_data(batch_size=BATCH_SIZE, smart_split=SMART_SPLIT)
    trainer = ModelTrainer(model=model,
                           optimizer=optimizer,
                           device=DEVICE,
                           nr_samples=NR_SAMPLES,
                           epochs=EPOCHS,
                           image_path=IMAGE_PATH,
                           reference_images=REFERENCE_IMAGES,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           test_dataloader=test_dataloader,
                           sampler=sampler)

    trainer.train()

    torch.save(trainer.best_model_checkpoint, os.path.join(MODEL_PATH, f"{RUN_NAME}_model.pth"))

    max_ssim = max(trainer.ssim_values)
    print(f"Max SSIM: {max_ssim}, At place: {5 * np.argmax(trainer.ssim_values)}")
    min_mae = min(trainer.mae_values)
    print(f"Min MAE: {min_mae}, At place: {5 * np.argmin(trainer.mae_values)}")

    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},train_losses.npz"), losses=trainer.train_losses)
    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},val_losses.npz"), losses=trainer.val_losses)
    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},ssim_values.npz"), losses=trainer.ssim_values)
    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},mae_values.npz"), losses=trainer.mae_values)

    plt.figure(figsize=(12, 6))
    plt.plot(trainer.train_losses[1:], label='Train Loss')
    plt.plot(trainer.val_losses[1:], label='Validation Loss')
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

    if GENERATE_IMAGES == True:
        model.load_state_dict(trainer.best_model_checkpoint)
        references_list, generated_list, structures_list = sample_model_output(model=model, sampler=sampler,
                                                                               n=len(test_dataloader) * BATCH_SIZE,
                                                                               test_dataloader=test_dataloader)
        save_image_list(references_list, REFERENCE_PATH)
        save_image_list(generated_list, SAMPLE_PATH)
        save_image_list(structures_list, STRUCTURE_PATH)