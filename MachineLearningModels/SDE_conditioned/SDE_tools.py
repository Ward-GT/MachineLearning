import torch
import numpy as np
from tqdm import tqdm
import logging
import torch.nn as nn
from SDE_utils import *
from losses import normal_kl, discretized_gaussian_log_likelihood

class DiffusionTools:
    def __init__(self, noise_steps: int, img_size: int, conditioned_prior: bool, vector_conditioning: bool, learn_sigma: bool, device: torch.DeviceObjType, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.conditioned_prior = conditioned_prior
        self.learn_sigma = learn_sigma
        self.vector_conditioning = vector_conditioning
        self.device = device

        self.prior_mean = None
        self.prior_variance = None

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(device)])

        assert self.alphas_cumprod_prev.shape == (self.noise_steps,)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([
            self.posterior_variance[1:2],
            self.posterior_variance[1:]
        ]))

        self.posterior_mean_coef_x0 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef_xt = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def prepare_noise_schedule(self):
        scale = 1000 / self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def init_prior_mean_variance(self, dataloader):
        all_images = []
        for i, (images, _, _) in enumerate(dataloader):
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)
        mean = torch.mean(all_images, dim=0)
        variance = torch.var(all_images, dim=0)

        self.prior_mean = mean
        self.prior_variance = variance
        print("Priors Initialized")

    def noise_images(self, x_start, t):

        sqrt_alpha_hat = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None, None]

        if self.conditioned_prior == True:
            if self.prior_mean == None or self.prior_variance == None:
                raise ValueError("Priors not initialized")
            else:
                mean = self.prior_to_batchsize(self.prior_mean, x_start.shape[0])
                variance = self.prior_to_batchsize(self.prior_variance, x_start.shape[0])
                assert mean.shape == variance.shape == x_start.shape
                # noise = torch.randn_like(x_start) * torch.sqrt(variance)
                noise = torch.randn_like(x_start)
                return (
                    sqrt_alpha_hat * (x_start-mean) + sqrt_one_minus_alpha_hat * noise, noise
                )

        noise = torch.randn_like(x_start)
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def get_specific_timesteps(self, timesteps, n):
        """
        Get a tensor of specific timesteps.

        Args:
            timesteps: List of timesteps to be converted to tensor.
            n: Batch size.

        Returns:
            Tensor of shape (n, len(timesteps)) on the correct device.
        """
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.long).to(self.device)
        return timesteps_tensor.repeat(n)

    def p_sample_loop(self, model, n, y):
        logging.info(f"Sampling {n} images")
        # if self.vector_conditioning == True:
        #     y = dimension_vectors_to_tensor(y)

        if y.shape[0] != n:
            y = concat_to_batchsize(y, n)

        if self.conditioned_prior == True:
            variance = self.prior_to_batchsize(self.prior_variance, n)
            mean = self.prior_to_batchsize(self.prior_mean, n)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            y = y.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                out = self.p_mean_variance(model, x, y, t)

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise

        model.train()
        if self.conditioned_prior == True:
            x = x + mean
        return x, y

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Calculate the posterior q(x_{t-1} | x_t, x_0)
        Args:
            x_start:
            x_t:
            t:

        Returns:

        """

        assert x_start.shape == x_t.shape

        posterior_mean = self.posterior_mean_coef_x0[t][:, None, None, None] * x_start + self.posterior_mean_coef_xt[t][:, None, None, None] * x_t
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None, None, None]

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean(self, x_t, t, eps):
        return (1 / torch.sqrt(self.alphas[t][:, None, None, None]) * (x_t - self.betas[t][:, None, None, None]*eps / torch.sqrt(1 - self.alphas_cumprod[t][:, None, None, None])))

    def p_mean_variance(self, model, x_t, y, t):
        """
        Calculate the predicted mean and variance p(x_{t-1} | x_t) --> eq (3)
        Args:
            model:
            x_t:
            y:
            t:

        Returns:

        """

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(x_t, y, t)
        if self.learn_sigma == True:
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            eps = model_output
            model_output = model_output.detach()

            min_log = self.posterior_log_variance_clipped[t][:, None, None, None]
            max_log = torch.log(self.betas[t][:, None, None, None])

            v = (model_var_values + 1) / 2
            model_log_variance = v*max_log + (1-v)*min_log
            model_variance = torch.exp(model_log_variance)
        else:
            eps = model_output
            model_variance = self.posterior_variance[t][:, None, None, None]
            model_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]

        model_mean = self.p_mean(x_t=x_t, t=t, eps=model_output)

        assert model_mean.shape == model_log_variance.shape == x_t.shape == eps.shape

        return {
            "mean": model_mean,
            "eps": eps,
            "variance": model_variance,
            "log_variance": model_log_variance,
        }

    def loss_vb(self, out, x_start, x_t, t):
        """
        Calculate the loss_vb term given by the kl divergence

        Args:
            model:
            x_start:
            x_t:
            y:
            t:
            clip_denoised:

        Returns:

        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        kl = normal_kl(mean1=true_mean, logvar1=true_log_variance_clipped, mean2=out["mean"], logvar2=out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        loss_vb = torch.where((t == 0), decoder_nll, kl)
        return loss_vb

    def training_losses(self, model, x_start, y, t):
        """
        Calculate the training losses for a single timestep
        Args:
            model:
            x_start:
            y:
            t:

        Returns:

        """
        # if self.vector_conditioning == True:
        #     y = dimension_vectors_to_tensor(y)

        x_start, y, t = x_start.to(self.device), y.to(self.device), t.to(self.device)

        x_t, noise = self.noise_images(x_start=x_start, t=t)
        mse = nn.MSELoss()

        terms = {}

        out = self.p_mean_variance(model, x_t, y, t)

        assert out["eps"].shape == noise.shape == x_start.shape

        if self.learn_sigma:
            terms["vb"] = self.loss_vb(out, x_start, x_t, t).mean()

        terms["mse"] = mse(noise, out["eps"])
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def prior_to_batchsize(self, prior, batchsize):
        return prior.unsqueeze(0).expand(batchsize, *prior.shape).to(self.device)