from BS_dataclass import get_data
from BS_model import UNet
from config import *
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import time
import logging
import os


def train():
    lossFunction = nn.BCEWithLogitsLoss()
    model = UNet().to(DEVICE)
    dataloader = get_data()
    opt = Adam(model.parameters(), lr=INIT_LR)

    logging.info(f"Starting training on {DEVICE}")
    startTime = time.time()
    for epoch in range(NUM_EPOCHS):
        logging.info(f"Starting epoch {epoch}:")
        model.train()
        totalTrainLoss = 0
        pbar = tqdm(dataloader)
        for i, (images, masks) in enumerate(pbar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = lossFunction(predictions, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(BCE=loss.item())

    endTime = time.time()
    logging.info(f"Training took {endTime - startTime} seconds")

    torch.save(model.state_dict(), MODEL_PATH)

train()