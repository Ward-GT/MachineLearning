import torch
import torch.nn.functional as F

IMG_size = 64 # Size of the images
BATCH_SIZE = 64 # Batch size
T = 300 # Number of steps in the diffusion process

# Pre-calculate different terms for closed form
start = 0.0001
end = 0.02
betas = torch.linspace(start, end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) #累积
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # 第一行添加1
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)