import os
import torch
import time
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from DDPM_utils import save_images, plot_images, get_data, calculate_FID
from DDPM_tools import DiffusionTools
from DDPM_model import UNet
from config import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

def train():
    device = DEVICE
    dataloader = get_data()
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR)
    mse = nn.MSELoss()
    diffusion = DiffusionTools(img_size=IMAGE_SIZE, device=device)
    logger = SummaryWriter(os.path.join("runs", RUN_NAME))
    l = len(dataloader)
    losses = []

    logging.info(f"Starting training on {device}")
    start_time = time.time()
    model.train()
    for epoch in range(EPOCHS):
        loss_total = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            loss_total += loss.item()

        average_loss = loss_total / len(dataloader)
        losses.append(average_loss)

        if epoch % 5 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join(IMAGE_PATH, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}_{epoch}.pth"))


    end_time = time.time()
    logging.info(f"Training took {end_time - start_time} seconds")
    np.savez(LOG_PATH, losses = losses)

    plt.figure(figsize=(12, 6))
    plt.plot(losses[1:], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

# train()