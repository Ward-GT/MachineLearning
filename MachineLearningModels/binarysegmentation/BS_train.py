import torch
from BS_dataclass import get_data
from BS_model import UNet
from config import *
from torch import nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level= logging.INFO, datefmt= "%I:%M:%S")

def train():
    lossFunction = nn.BCEWithLogitsLoss()
    model = UNet().to(DEVICE)
    train_dataloader, test_dataloader = get_data()
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    training_losses = []
    testing_losses = []

    logging.info(f"Starting training on {DEVICE}")
    startTime = time.time()

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Starting epoch {epoch}:")

        model.train()
        train_loss_total = 0
        pbar = tqdm(train_dataloader)
        for i, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            train_loss = lossFunction(predictions, masks)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            pbar.set_postfix(BCE_train=train_loss.item())

            train_loss_total += train_loss.item()

        average_train_loss = train_loss_total / len(train_dataloader)
        training_losses.append(average_train_loss)

        model.eval()
        test_loss_total = 0

        with torch.no_grad():
            for images, masks in test_dataloader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                predictions = model(images)
                test_loss = lossFunction(predictions, masks)

                test_loss_total += test_loss.item()

        average_test_loss = test_loss_total / len(test_dataloader)
        logging.info("Train loss: " + str(average_train_loss) + ", Test loss: " + str(average_test_loss))
        testing_losses.append(average_test_loss)

        scheduler.step(average_test_loss)

    endTime = time.time()
    logging.info(f"Training took {endTime - startTime} seconds")

    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(LOG_PATH, training_losses=training_losses, testing_losses=testing_losses)

    plt.figure(figsize=(12, 6))
    plt.plot(training_losses[1:], label='Train Loss')
    plt.plot(testing_losses[1:], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.show()

train()