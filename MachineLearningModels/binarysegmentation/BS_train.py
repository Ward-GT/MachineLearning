from BS_dataclass import get_data
from BS_model import UNet
from config import *
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import time
import os


def train():
    lossFunction = nn.BCEWithLogitsLoss()
    model = UNet().to(DEVICE)
    dataloader = get_data()
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
            loss = lossFunction(predictions, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")

    endTime = time.time()
    print(f"Training took {endTime - startTime} seconds")

    torch.save(model.state_dict(), MODEL_PATH)

train()