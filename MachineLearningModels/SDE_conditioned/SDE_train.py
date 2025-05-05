import copy
import os
import torch
import time
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from SDE_test import sample_model_output
from SDE_utils import *
from SDE_tools import DiffusionTools
from SDE_test import calculate_metrics
from itertools import cycle

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            optimizer: optim,
            image_path: str,
            model_path: str,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            diffusion: DiffusionTools,
            **kwargs
    ):
        super().__init__()

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.device = device
        self.mse = nn.MSELoss()
        self.nr_samples = 5
        self.ema = kwargs.get("ema")
        self.epochs = kwargs.get("epochs")
        self.image_path = image_path
        self.model_path = model_path
        self.threshold_training = kwargs.get("threshold_training")
        self.threshold = kwargs.get("threshold")
        self.ema_decay = kwargs.get("ema_decay")
        self.clip_grad = kwargs.get("clip_grad")
        self.vector_conditioning = kwargs.get("vector_conditioning")
        self.mixed_precision = kwargs.get("mixed_precision", False)
        self.batch_size = kwargs.get("batch_size")
        self.val_samples = 100

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(device.type) if self.mixed_precision else None

        self.train_losses = []
        self.val_losses = []
        self.ssim_values = []
        self.mae_values = []

        self.best_val_loss = float('inf')
        self.best_model_checkpoint = None
        self.best_model_epoch = None

        self.ema_model = copy.deepcopy(model)
        self.ema_model.to(self.device)
        for param in self.ema_model.parameters():
            param.detach()

    def train_epoch(self):
        loss_total = 0
        self.model.train()
        pbar = tqdm(self.train_dataloader)
        data_time = 0
        train_time = 0
        data_start_time = time.time()
        for i, (images, structures, _, vectors) in enumerate(pbar):
            data_end_time = time.time()
            y = (vectors if self.vector_conditioning else structures)
            t = self.diffusion.sample_timesteps(images.shape[0])
            train_start_time = time.time()
            # Mixed precision training logic
            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast(device_type=self.device.type, enabled=True, dtype=torch.float16):
                    losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
                    loss = losses["loss"]

                # Scale loss and perform backward pass
                self.scaler.scale(loss).backward()

                # Unscale gradients and optimize
                if self.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training if mixed precision is off
                losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
                loss = losses["loss"]
                loss.backward()
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            if self.ema:
                self.update_ema()

            train_end_time = time.time()

            loss_total += loss.item()
            pbar.set_postfix(Loss=loss.item())

            data_time += (data_end_time - data_start_time)
            train_time += (train_end_time - train_start_time)
            data_start_time = time.time()

        average_loss = loss_total / len(self.train_dataloader)
        self.train_losses.append(average_loss)
        logging.info(f"Data time: {data_time}, Train time: {train_time}")

    def validation_epoch(self):
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader)

            for i, (images, structures, _, vectors) in enumerate(pbar):
                y = (vectors if self.vector_conditioning else structures)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                # Use autocast for validation if mixed precision is on
                if self.mixed_precision:
                    with autocast(device_type=self.device.type, enabled=True, dtype=torch.float16):
                        losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
                        loss = losses["loss"].mean()
                else:
                    losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
                    loss = losses["loss"].mean()

                pbar.set_postfix(loss=loss.item())
                loss_total += loss.item()

            average_loss = loss_total / len(self.val_dataloader)
            self.val_losses.append(average_loss)
            return average_loss

    def generate_reference_images(self, epoch):
        n = self.batch_size if epoch < 0.7 * self.epochs else len(self.val_dataloader.dataset)

        if self.ema == True:
            references_list, generated_list, structures_list = sample_model_output(model=self.ema_model,
                                                                                   device=self.device,
                                                                                   sampler=self.diffusion,
                                                                                   dataloader=self.val_dataloader,
                                                                                   n=n,
                                                                                   batch_size=self.batch_size)
        else:
            references_list, generated_list, structures_list = sample_model_output(model=self.model,
                                                                                   device=self.device,
                                                                                   sampler=self.diffusion,
                                                                                   dataloader=self.val_dataloader,
                                                                                   n=n,
                                                                                   batch_size=self.batch_size)

        ssim, _, mae, _ = calculate_metrics(references_list, generated_list)
        save_images(reference_images=references_list[:5], generated_images=generated_list[:5],
                    structure_images=structures_list[:5], path=os.path.join(self.image_path, f"{epoch}.jpg"))
        return np.mean(ssim), np.mean(mae)

    def update_ema(self):
        self.model.eval()
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def update_best_model(self, epoch):
        self.best_model_checkpoint = self.model.state_dict()
        self.best_model_epoch = epoch
        torch.save(self.best_model_checkpoint, os.path.join(self.model_path, "best_model.pth"))

    def train(self):
        logging.info(f"Starting training on {self.device}")

        # Mixed precision setup note
        if self.mixed_precision:
            logging.info("Mixed precision training enabled")

        self.model.to(self.device)
        start_time = time.time()
        if self.threshold_training == False:
            for epoch in range(self.epochs):
                logging.info(f"Starting epoch {epoch}:")

                logging.info("Starting train loop")
                self.train_epoch()

                logging.info("Starting validation loop")
                val_loss = self.validation_epoch()

                if val_loss < self.best_val_loss and self.ema == True:
                    self.best_val_loss = val_loss
                    self.update_best_model(epoch)

                if (epoch + 1) % 5 == 0:
                    ssim, mae = self.generate_reference_images(epoch)

                    if self.ema == False and mae < self.best_val_loss:
                        self.best_val_loss = mae
                        self.update_best_model(epoch)

                    self.ssim_values.append(ssim)
                    self.mae_values.append(mae)

        elif self.threshold_training == True:
            mae = float('inf')
            epoch = 0

            while mae > self.threshold:
                logging.info(f"Starting epoch {epoch}:")

                logging.info("Starting train loop")
                self.train_epoch()

                logging.info("Starting validation loop")
                val_loss = self.validation_epoch()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_checkpoint = self.model.state_dict()
                    self.best_model_epoch = epoch

                if (epoch + 1) % 5 == 0:
                    ssim, mae = self.generate_reference_images(epoch)
                epoch += 1

        end_time = time.time()
        logging.info(f"Training took {end_time - start_time} seconds")