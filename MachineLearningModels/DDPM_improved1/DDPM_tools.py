import os
import torch
import time
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from DDPM_utils import save_images, plot_images, setup_logging, get_data
from DDPM_model import UNet
from config import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

class DiffusionTools:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alphas_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} images")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x- ((1 - alpha)/ (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train():
    setup_logging(RUN_NAME)
    device = DEVICE
    dataloader = get_data()
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR)
    mse = nn.MSELoss()
    diffusion = DiffusionTools(img_size=IMAGE_SIZE, device=device)
    logger = SummaryWriter(os.path.join("runs", RUN_NAME))
    l = len(dataloader)

    logging.info(f"Starting training on {device}")
    start_time = time.time()
    model.train()
    for epoch in range(EPOCHS):
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

        if epoch % 5 == 0:
            sampled_images = diffusion.sample(model, n=1)
            save_images(sampled_images, os.path.join(RESULT_PATH, RUN_NAME, f"{epoch}.jpg"))

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{RUN_NAME}.pth"))
    end_time = time.time()
    logging.info(f"Training took {end_time - start_time} seconds")

#train()
