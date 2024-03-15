from BS_dataclass import SegmentationDataset
from BS_model import UNet
from config import *
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import os


def train(model, dataloader):
    lossFunction = nn.BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=INIT_LR)

    print ("[INFO] Starting training")
    startTime = time.time()
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        totalTrainLoss = 0
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = lossFunction(predictions, masks.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {totalTrainLoss/len(dataloader)}")

    endTime = time.time()
    print(f"Training took {endTime - startTime} seconds")

    torch.save(model.state_dict(), MODEL_PATH)

