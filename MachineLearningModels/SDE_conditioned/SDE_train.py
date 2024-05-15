import os
import torch
import time
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from SDE_SimpleUNet import SimpleUNet
from SDE_UNet import UNet
from SDE_utils import *
from SDE_tools import DiffusionTools
from SDE_test import sample_model_output
from SDE_datareduction import get_data
from itertools import cycle
from config import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

def train():
    set_seed()
    device = DEVICE
    nr_samples = NR_SAMPLES
    train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(smart_split=SMART_SPLIT)
    model = UNet(n_blocks=N_BLOCKS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    diffusion = DiffusionTools()
    train_losses = []
    val_losses = []

    logging.info(f"Starting training on {device}")
    start_time = time.time()
    for epoch in range(EPOCHS):
        loss_total = 0
        logging.info(f"Starting epoch {epoch}:")
        logging.info("Starting train loop")
        model.train()
        pbar = tqdm(train_dataloader)
        for i, (images, structures, _) in enumerate(pbar):
            images, structures = images.to(device), structures.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            x_t_struct = concatenate_images(x_t, structures)
            predicted_noise = model(x_t_struct, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            loss_total += loss.item()

        average_loss = loss_total / len(train_dataloader)
        train_losses.append(average_loss)

        loss_total = 0
        logging.info("Starting validation loop")
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            for i, (images, structures, _) in enumerate(pbar):
                images, structures = images.to(device), structures.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                x_t_struct = concatenate_images(x_t, structures)
                predicted_noise = model(x_t_struct, t)
                loss = mse(noise, predicted_noise)

                pbar.set_postfix(MSE=loss.item())
                loss_total += loss.item()

        average_loss = loss_total / len(val_dataloader)
        val_losses.append(average_loss)

        if (epoch+1) % 5 == 0:
            if REFERENCE_IMAGES == True:
                test_images, test_structures, _ = next(cycle(test_dataloader))
                test_images = concat_to_batchsize(test_images, nr_samples)
                test_images = tensor_to_PIL(test_images)
                test_structures = test_structures.to(device)
                sampled_images, structures = diffusion.sample(model, n=nr_samples, structures=test_structures)
                save_images(reference_images=test_images, generated_images=sampled_images, structure_images=structures, path=os.path.join(IMAGE_PATH, f"{epoch}.jpg"))

            if epoch > 0.9*EPOCHS:
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}_{epoch}.pth"))

    end_time = time.time()
    logging.info(f"Training took {end_time - start_time} seconds")

    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},train_losses.npz"), losses = train_losses)
    np.savez(os.path.join(RESULT_PATH, f"{RUN_NAME},val_losses.npz"), losses = val_losses)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}_final.pth"))

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses[1:], label='Train Loss')
    plt.plot(val_losses[1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(RESULT_PATH, "losses.jpg"))
    plt.show()

    if GENERATE_IMAGES == True:
        references_list, generated_list, structures_list = sample_model_output(model=model, sampler=diffusion, n=len(test_dataloader) * BATCH_SIZE, test_dataloader=test_dataloader)
        save_image_list(references_list, REFERENCE_PATH)
        save_image_list(generated_list, SAMPLE_PATH)
        save_image_list(structures_list, STRUCTURE_PATH)

train()