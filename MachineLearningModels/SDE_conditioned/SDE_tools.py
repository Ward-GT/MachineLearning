from tqdm import tqdm
import logging
from config import *
from SDE_utils import *

class DiffusionTools:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=1e-4, beta_end=0.02, img_size=IMAGE_SIZE, device=DEVICE):
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

    def sample(self, model, n, structures):
        logging.info(f"Sampling {n} images")
        if structures.shape[0] != n:
            structures = concat_to_batchsize(structures, n)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            structures.to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                x_struct = concatenate_images(x, structures)
                predicted_noise = model(x_struct, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x- ((1 - alpha)/ (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = tensor_to_PIL(x)
        structures = tensor_to_PIL(structures)
        return x, structures



