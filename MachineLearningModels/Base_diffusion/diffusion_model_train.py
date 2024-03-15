import torch.nn.functional as F
from torch.optim import Adam
import torch
from diffusion_model_sampler import forward_diffusion_sample, sample_plot_image
from diffusion_parameters import *

def get_loss(model, x_0, t, device = "cpu"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    print(f"Noise shape: {noise.shape}")
    print(f"Noise_pred shape: {noise_pred.shape}")
    return F.l1_loss(noise, noise_pred)

def train_diffusion_model(model, dataloader, epochs):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on {device}")
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch, X in enumerate(dataloader):
            optimizer.zero_grad()
            print(f"X shape: {X.shape}")
            print(f"Batch: {batch}")
            t = torch.randint(0, T, (X.shape[0],), device=device).long()
            loss = get_loss(model, X, t, device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and batch == 0:
                print(f"Epoch {epoch} | step {batch:03d} Loss: {loss.item()} ")
                sample_plot_image(model, device)
    print("Training finished")

t = torch.randint(0, T, (BATCH_SIZE,), device="cpu").long()
print(t.shape)
print(t[0])