import torch
import numpy as np
from tqdm import tqdm
import logging
import torch.nn as nn
from SDE_utils import *
from losses import normal_kl, discretized_gaussian_log_likelihood

class GaussianDiffusion:
    def __init__(self, noise_steps: int, image_size: int, device: torch.device, learn_sigma: bool, conditioned_prior: bool = False, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        self.device = device
        self.learn_sigma = learn_sigma
        self.conditioned_prior = conditioned_prior

        self.prior_mean = None
        self.prior_variance = None

        self.betas = self.prepare_noise_schedule().astype(np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.noise_steps,)

        # calculations for diffusion (q(x_t) | x_{t-1}) --> eq (2)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0) --> eq (12)
        self.posterior_variance = self.betas*(1.0-self.alphas_cumprod_prev) / (1.0-self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef_x0 = self.betas*np.sqrt(self.alphas_cumprod_prev)/(1-self.alphas_cumprod)
        self.posterior_mean_coef_xt = np.sqrt(self.alphas)*(1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)

    def prepare_noise_schedule(self):
        scale = 1000 / self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return np.linspace(beta_start, beta_end, self.noise_steps, dtype=np.float64)

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
        return timesteps_tensor.unsqueeze(0).expand(n, -1)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0) --> eq (8)
        Args:
            x_start: starting sample
            t: timestep

        Returns:
            mean: the mean of the gaussian of the distribution
            variance: the variance of the gaussian of the distribution
            log_variance: the log of the variance

        """
        mean = self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)*x_start
        variance = self.extract_into_tensor((1.0-self.alphas_cumprod), t, x_start.shape)
        log_variance = self.extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return mean, variance, log_variance

    def noise_images(self, x_start, t):
        """
        Noise the images for a given number of diffusion steps

        Args:
            x_start: The data sample to sample from
            t: The timesteps to sample

        Returns:
            The noisified image and the noise that was added to the image
        """

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
                    self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * (x_start - mean) +
                    self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise, noise
                )
        else:
            noise = torch.randn_like(x_start)
            return (
                self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise, noise
            )

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

        posterior_mean = (
                self.extract_into_tensor(self.posterior_mean_coef_x0, t, x_t.shape)*x_start +
                self.extract_into_tensor(self.posterior_mean_coef_xt, t, x_t.shape)*x_t
        )

        posterior_variance = self.extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean(self, x_t, t, eps):
        return (1 / torch.sqrt(self.extract_into_tensor(self.alphas, t, x_t.shape)) *
                (x_t - self.extract_into_tensor(self.betas, t, x_t.shape)*eps / self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)))

    def p_mean_variance(self, model, x_t, y, t, clip_denoised = True):
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

            min_log = self.extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = self.extract_into_tensor(np.log(self.betas), t, x_t.shape)

            v = (model_var_values + 1) / 2
            model_log_variance = v*max_log + (1-v)*min_log
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = self.extract_into_tensor(np.append(self.posterior_variance[1], self.betas[1:]), t, x_t.shape)
            model_log_variance = self.extract_into_tensor(np.log(np.append(self.posterior_variance[1], self.betas[1:])), t, x_t.shape)

        pred_xstart = self.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)

        if clip_denoised == True:
            pred_xstart = pred_xstart.clamp(-1, 1)

        # model_mean, _, _= self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        model_mean = self.p_mean(x_t=x_t, t=t, eps=model_output)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def predict_xstart_from_eps(self, x_t, t, eps):
        """
        Predict the start image from eps --> eq (4) DDPM
        Args:
            x_t:
            t:
            eps:

        Returns:

        """
        assert x_t.shape == eps.shape
        return (
                self.extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self.extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, model, x, y, t, clip_denoised=True):
        """
        Sample x_{t-1} from the model at the given timestep

        Args:
            model:
            x:
            y:
            t:
            clip_denoised:

        Returns:

        """

        out = self.p_mean_variance(model, x, y, t, clip_denoised=clip_denoised)
        if self.conditioned_prior == True:
            # variance = self.prior_to_batchsize(self.prior_variance, x.shape[0])
            # noise = torch.randn_like(x) * torch.sqrt(variance)
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1))) # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, n, y):
        logging.info(f"Sampling {n} images")
        if y.shape[0] != n:
            y = concat_to_batchsize(y, n)

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            y.to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, y, t)
                alpha = self.extract_into_tensor(self.alphas, t, x.shape)
                alpha_hat = self.extract_into_tensor(self.alphas_cumprod, t, x.shape)
                beta = self.extract_into_tensor(self.betas, t, x.shape)

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

        model.train()
        return x, y

    # def p_sample_loop(self, model, n, y, clip_denoised=True):
    #     """
    #     Generate samples from the model
    #     Args:
    #         model:
    #         n:
    #         y:
    #         clip_denoised:
    #
    #     Returns:
    #
    #     """
    #
    #     logging.info(f"Sampling {n} images")
    #     if y.shape[0] != n:
    #         y = concat_to_batchsize(y, n)
    #
    #     assert y.shape[0] == n
    #
    #     model.eval()
    #     with torch.no_grad():
    #         if self.conditioned_prior == True:
    #             # variance = self.prior_to_batchsize(self.prior_variance, n)
    #             # x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device) * torch.sqrt(variance)
    #             x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
    #         else:
    #             x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
    #         y.to(self.device)
    #
    #         for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
    #             t = (torch.ones(n) * i).long().to(self.device)
    #             out = self.p_sample(model=model, x=x, y=y, t=t, clip_denoised=clip_denoised)
    #             x = out["sample"]
    #
    #     if self.conditioned_prior == True:
    #         mean = self.prior_to_batchsize(self.prior_mean, n)
    #         x += mean
    #
    #     return x, y

    def loss_vb(self, model, x_start, x_t, t, y=None, clip_denoised=True):
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
        out = self.p_mean_variance(model=model, x_t=x_t, y=y, t=t, clip_denoised=clip_denoised)
        kl = normal_kl(mean1=true_mean, logvar1=true_log_variance_clipped, mean2=out["mean"], logvar2=out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

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
        x_t, noise = self.noise_images(x_start=x_start, t=t)
        mse = nn.MSELoss()

        terms = {}

        model_output = model(x_t, y, t)

        if self.learn_sigma == True:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
            terms["vb"] = self.loss_vb(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
            )["output"]
            terms["vb"] *= self.noise_steps / 1000.0

        assert model_output.shape == noise.shape == x_start.shape
        epsilon = 0.0001
        # if self.conditioned_prior == True:
        #     variance = self.prior_to_batchsize(self.prior_variance, x_start.shape[0])
        #     variance = variance + epsilon
        #     terms["mse"] = mean_flat(((noise - model_output) * variance ) ** 2)
        # else:
        #     terms["mse"] = mean_flat((noise - model_output) ** 2)
        terms["mse"] = mse(noise, model_output)
        if self.learn_sigma == True:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def extract_into_tensor(self, arr, t, broadcast_shape):
        res = torch.from_numpy(arr).to(device=self.device)[t].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def prior_to_batchsize(self, prior, batchsize):
        return prior.unsqueeze(0).expand(batchsize, *prior.shape).to(self.device)

class DiffusionTools:
    def __init__(self, noise_steps: int, img_size: int, conditioned_prior: bool, vector_conditioning: bool, device: torch.DeviceObjType, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.conditioned_prior = conditioned_prior
        self.learn_sigma = False
        self.vector_conditioning = vector_conditioning
        self.device = device

        self.prior_mean = None
        self.prior_variance = None

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

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

        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alphas_hat[t])[:, None, None, None]

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
                predicted_noise = model(x, y, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alphas_hat[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                    # if self.conditioned_prior == True:
                    #     noise = noise * torch.sqrt(variance)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x- ((1 - alpha)/ (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        if self.conditioned_prior == True:
            x = x + mean
        return x, y

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
        model_output = model(x_t, y, t)

        assert model_output.shape == noise.shape == x_start.shape

        terms["mse"] = mse(noise, model_output)
        terms["loss"] = terms["mse"]

        return terms

    def prior_to_batchsize(self, prior, batchsize):
        return prior.unsqueeze(0).expand(batchsize, *prior.shape).to(self.device)