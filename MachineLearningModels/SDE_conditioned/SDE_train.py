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
            ema: bool = True,
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
        self.ema = ema
        self.epochs = kwargs.get("epochs")
        self.image_path = image_path
        self.model_path = model_path
        self.threshold_training = kwargs.get("threshold_training")
        self.threshold = kwargs.get("threshold")
        self.ema_decay = kwargs.get("ema_decay")
        self.clip_grad = kwargs.get("clip_grad")
        self.vector_conditioning = kwargs.get("vector_conditioning")

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

        for i, (images, structures, vectors) in enumerate(pbar):
            y = (vectors if self.vector_conditioning else structures)
            t = self.diffusion.sample_timesteps(images.shape[0])
            losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
            loss = losses["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            if self.ema == True:
                self.update_ema()

            loss_total += loss.item()
            pbar.set_postfix(MSE=loss.item())

        average_loss = loss_total / len(self.train_dataloader)
        self.train_losses.append(average_loss)

    def validation_epoch(self):
        loss_total = 0
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_dataloader)

            for i, (images, structures, vectors) in enumerate(pbar):
                y = (vectors if self.vector_conditioning else structures)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                losses = self.diffusion.training_losses(model=self.model, x_start=images, y=y, t=t)
                loss = losses["loss"].mean()

                pbar.set_postfix(MSE=loss.item())
                loss_total += loss.item()

            average_loss = loss_total / len(self.val_dataloader)
            self.val_losses.append(average_loss)
            return average_loss

    def generate_reference_images(self, epoch):
        test_images, test_structures, test_vectors = next(cycle(self.test_dataloader))
        test_images = concat_to_batchsize(test_images, self.nr_samples)
        test_structures = concat_to_batchsize(test_structures, self.nr_samples)
        y = (test_vectors if self.vector_conditioning else test_structures)
        if self.ema == True:
            sampled_images, structures = self.diffusion.p_sample_loop(self.ema_model, n=self.nr_samples, y=y)
        else:
            sampled_images, _ = self.diffusion.p_sample_loop(self.model, n=self.nr_samples, y=y)
        test_images = tensor_to_PIL(test_images)
        sampled_images = tensor_to_PIL(sampled_images)
        structures = tensor_to_PIL(test_structures)
        ssim, _, _, _, mae = calculate_metrics(sampled_images, test_images)
        save_images(reference_images=test_images, generated_images=sampled_images,
                    structure_images=structures, path=os.path.join(self.image_path, f"{epoch}.jpg"))
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
        self.model.to(self.device)
        if self.diffusion.conditioned_prior == True:
            self.diffusion.init_prior_mean_variance(self.train_dataloader)
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