import torch
import os
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from SDE_train import ModelTrainer
from SDE_UNet import UNet
from SDE_datareduction import get_data
from SDE_tools import DiffusionTools
from SDE_test import sample_model_output, calculate_metrics, sample_save_metrics
from SDE_utils import save_image_list, set_seed, load_images

# Base Paths
BASE_OUTPUT = "results"
# BASE_OUTPUT = r"E:\Ward Taborsky\results"

BASE_INPUT = r"C:\Users\20202137\Documents\Python\MachineLearning\data"
# BASE_INPUT = r"E:\Ward Taborsky"
# BASE_INPUT = r"/home/tue/20234635/MachineLearningGit/MachineLearningModels/data"

# Dataset paths
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_specific")
DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined")
# DATASET_PATH = os.path.join(BASE_INPUT, "figure_B_combined_small")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "Output")
STRUCTURE_DATASET_PATH = os.path.join(DATASET_PATH, "Structure")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE == "cuda" else False

# Training settings
TRAINING = True
SMART_SPLIT = False
REFERENCE_IMAGES = True
GENERATE_IMAGES = True

# Test settings
TESTING = False
CALCULATE_METRICS = False
SAMPLE_METRICS = False
TEST_PATH = r"C:\Users\20202137\OneDrive - TU Eindhoven\Jaar 4\BEP\Results\Results SDE_conditioned\UNet_nblocks_2_smartsplit_False_split_0.5_imgsize_128_epochs_1000"
MODEL_NAME = r"UNet_nblocks_2_smartsplit_False_split_0.5_imgsize_128_epochs_1000_final.pth"

# Training parameters
TEST_SPLIT = 0.5
VALIDATION_SPLIT = 0.1
EPOCHS = 500
BATCH_SIZE = 20
IMAGE_SIZE = 64
INIT_LR = 0.00002
WEIGHT_DECAY = 0.001
DEFAULT_SEED = 42

# Sampling parameters
NOISE_STEPS = 1000
NR_SAMPLES = 5

# UNet Parameters
MODEL = "UNet"
N_BLOCKS = 1
TIME_EMB_DIM = 128

RUN_NAME = f"{MODEL}_nblocks_{N_BLOCKS}_smartsplit_{SMART_SPLIT}_split_{TEST_SPLIT}_imgsize_{IMAGE_SIZE}_epochs_{EPOCHS}"

if TRAINING:
    # Output paths
    RESULT_PATH = os.path.join(BASE_OUTPUT, RUN_NAME)
    MODEL_PATH = os.path.join(RESULT_PATH, "models")
    IMAGE_PATH = os.path.join(RESULT_PATH, "images")
    SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
    REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
    STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")

    set_seed(seed=DEFAULT_SEED)
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
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    sampler = DiffusionTools(noise_steps=NOISE_STEPS, img_size=IMAGE_SIZE, device=DEVICE)
    train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(batch_size=BATCH_SIZE, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT, image_size=IMAGE_SIZE,
                                                                          image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH, result_path=RESULT_PATH,
                                                                          smart_split=SMART_SPLIT)
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
        model.load_state_dict(trainer.best_model_checkpoint)
        references_list, generated_list, structures_list = sample_model_output(model=model, sampler=sampler, device=DEVICE,
                                                                               n=len(test_dataloader) * BATCH_SIZE, batch_size=BATCH_SIZE,
                                                                               test_dataloader=test_dataloader)
        save_image_list(references_list, REFERENCE_PATH)
        save_image_list(generated_list, SAMPLE_PATH)
        save_image_list(structures_list, STRUCTURE_PATH)

if TESTING:
    MODEL_PATH = os.path.join(os.path.join(TEST_PATH, "models"), MODEL_NAME)
    IMAGE_PATH = os.path.join(TEST_PATH, "images")
    SAMPLE_PATH = os.path.join(IMAGE_PATH, "Samples")
    REFERENCE_PATH = os.path.join(IMAGE_PATH, "References")
    STRUCTURE_PATH = os.path.join(IMAGE_PATH, "Structures")
    TEST_DATASET_PATH = os.path.join(TEST_PATH, "test_indices.pth")

    if CALCULATE_METRICS == True:
        structure_images = load_images(STRUCTURE_PATH)
        reference_images = load_images(REFERENCE_PATH)
        sampled_images = load_images(SAMPLE_PATH)
        ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values = calculate_metrics(reference_images, sampled_images)
        print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MAE: {np.mean(mae_values)}, MSE Mean: {np.mean(mse_mean_values)}, MSE Max: {np.mean(mse_max_values)}")

    if SAMPLE_METRICS == True:
        set_seed(seed=DEFAULT_SEED)
        model = UNet(n_blocks=N_BLOCKS)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(DEVICE)
        sampler = DiffusionTools(noise_steps=NOISE_STEPS, img_size=IMAGE_SIZE, device=DEVICE)
        sample_save_metrics(model=model, sampler=sampler, image_size=IMAGE_SIZE, device=DEVICE,
                            image_dataset_path=IMAGE_DATASET_PATH, structure_dataset_path=STRUCTURE_DATASET_PATH, test_path=TEST_DATASET_PATH, reference_path=REFERENCE_PATH, sample_path=SAMPLE_PATH, structure_path=STRUCTURE_PATH,
                            n=NR_SAMPLES, batch_size=BATCH_SIZE)

