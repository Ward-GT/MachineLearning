import os
import torch
import time
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from SDE_SimpleUNet import UNet
from SDE_utils import *
from SDE_tools import DiffusionTools
from itertools import cycle
from config import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")


def train():
    device = DEVICE
    nr_samples = 5
    train_dataloader, test_dataloader = get_data()
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    diffusion = DiffusionTools()
    losses = []

    logging.info(f"Starting training on {device}")
    start_time = time.time()
    model.train()
    for epoch in range(EPOCHS):
        loss_total = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, structures) in enumerate(pbar):
            images = images.to(device)
            structures = structures.to(device)
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
        losses.append(average_loss)

        if epoch % 5 == 0:
            test_images, test_structures = next(cycle(test_dataloader))
            test_images = concat_to_batchsize(test_images, nr_samples)
            test_images = tensor_to_PIL(test_images)
            test_structures = test_structures.to(device)
            sampled_images, structures = diffusion.sample(model, n=nr_samples, structures=test_structures)
            save_images(reference_images=test_images, generated_images=sampled_images, structure_images=structures, path=os.path.join(IMAGE_PATH, f"{epoch}.jpg"))
            if epoch > 0.9*EPOCHS:
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}_{epoch}.pth"))

    end_time = time.time()
    logging.info(f"Training took {end_time - start_time} seconds")
    np.savez(LOG_PATH, losses = losses)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}_final.pth"))
    plt.figure(figsize=(12, 6))
    plt.plot(losses[1:], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(RESULT_PATH, "loss.jpg"))
    plt.show()

train()