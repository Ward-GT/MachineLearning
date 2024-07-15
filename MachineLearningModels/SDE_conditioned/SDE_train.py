import os
import torch
import time
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from SDE_utils import *
from SDE_tools import DiffusionTools
from SDE_test import sample_model_output, calculate_metrics
from itertools import cycle

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim,
                 device: torch.device,
                 nr_samples: int,
                 epochs: int,
                 image_path: str,
                 reference_images: bool,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 sampler: DiffusionTools):
        super().__init__()

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.sampler = sampler
        self.device = device
        self.mse = nn.MSELoss()
        self.nr_samples = nr_samples
        self.epochs = epochs
        self.image_path = image_path
        self.reference_images = reference_images

        self.train_losses = []
        self.val_losses = []
        self.ssim_values = []
        self.mae_values = []

        self.best_val_loss = float('inf')
        self.best_model_checkpoint = None

    def train_epoch(self):
        loss_total = 0
        self.model.train()
        pbar = tqdm(self.train_dataloader)

        for i, (images, structures, _) in enumerate(pbar):
            images, structures = images.to(self.device), structures.to(self.device)
            t = self.sampler.sample_timesteps(images.shape[0]).to(self.device)
            x_t, noise = self.sampler.noise_images(images, t)
            x_t_struct = concatenate_images(x_t, structures)
            predicted_noise = self.model(x_t_struct, t)
            loss = self.mse(noise, predicted_noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            pbar.set_postfix(MSE=loss.item())

        average_loss = loss_total / len(self.train_dataloader)
        self.train_losses.append(average_loss)

    def validation_epoch(self):
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader)

            for i, (images, structures, _) in enumerate(pbar):
                images, structures = images.to(self.device), structures.to(self.device)
                t = self.sampler.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.sampler.noise_images(images, t)
                x_t_struct = concatenate_images(x_t, structures)
                predicted_noise = self.model(x_t_struct, t)
                loss = self.mse(noise, predicted_noise)

                pbar.set_postfix(MSE=loss.item())
                loss_total += loss.item()

            average_loss = loss_total / len(self.val_dataloader)
            self.val_losses.append(average_loss)
            return average_loss

    def generate_reference_images(self, epoch):
        test_images, test_structures, _ = next(cycle(self.test_dataloader))
        test_images = concat_to_batchsize(test_images, self.nr_samples)
        test_structures = test_structures.to(self.device)
        sampled_images, structures = self.sampler.sample(self.model, n=self.nr_samples, structures=test_structures)
        test_images = tensor_to_PIL(test_images)
        sampled_images = tensor_to_PIL(sampled_images)
        structures = tensor_to_PIL(structures)
        ssim, _, _, _, mae = calculate_metrics(sampled_images, test_images)
        self.ssim_values.append(np.mean(ssim))
        self.mae_values.append(np.mean(mae))
        save_images(reference_images=test_images, generated_images=sampled_images,
                    structure_images=structures, path=os.path.join(self.image_path, f"{epoch}.jpg"))
        return np.mean(ssim), np.mean(mae)

    def train(self):
        logging.info(f"Starting training on {self.device}")
        self.model.to(self.device)
        start_time = time.time()
        for epoch in range(self.epochs):
            logging.info(f"Starting epoch {epoch}:")

            logging.info("Starting train loop")
            self.train_epoch()

            logging.info("Starting validation loop")
            val_loss = self.validation_epoch()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_checkpoint = self.model.state_dict()

            if (epoch + 1) % 5 == 0:
                if self.reference_images == True:
                    ssim, mae = self.generate_reference_images(epoch)

        end_time = time.time()
        logging.info(f"Training took {end_time - start_time} seconds")