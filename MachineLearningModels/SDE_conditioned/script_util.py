import math
from models.SR3_UNet import UNet
from models.SDE_UNet import MiddleUNet
from models.SDE_SimpleUNet import SimpleUNet
from SDE_tools import DiffusionTools

def create_model_diffusion(device, **kwargs):
    model = create_model(
        model_name=kwargs.get("model_name"),
        image_size=kwargs.get("image_size"),
        learn_sigma=kwargs.get("learn_sigma"),
        n_blocks=kwargs.get("n_blocks"),
        n_heads=kwargs.get("n_heads"),
        dim_head=kwargs.get("dim_head"),
        n_channels=kwargs.get("n_channels"),
        attention_resolutions=kwargs.get("attention_resolutions"),
        vector_conditioning=kwargs.get("vector_conditioning"),
        device=device
    )

    diffusion = create_diffusion(
        noise_steps=kwargs.get("noise_steps"),
        image_size=kwargs.get("image_size"),
        device=device,
        learn_sigma=kwargs.get("learn_sigma"),
        conditioned_prior=kwargs.get("conditioned_prior"),
        vector_conditioning=kwargs.get("vector_conditioning")
    )

    return model, diffusion

def create_model(
        model_name,
        image_size,
        learn_sigma,
        n_blocks,
        n_heads,
        dim_head,
        n_channels,
        attention_resolutions,
        vector_conditioning,
        device
):

    if model_name == "SimpleUNet":
        return SimpleUNet(
            n_channels=n_channels,
            image_channels=6,
            out_dim=(3 if not learn_sigma else 6)
        ).to(device)

    elif model_name == "UNet":
        if image_size == 256:
            channel_mult = [1, 1, 2, 2, 4, 4]
        elif image_size == 128:
            channel_mult = [1, 2, 2, 3, 4]
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")

        attention_ds = []

        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        is_attn = [False for _ in range(len(channel_mult))]

        for res in attention_ds:
            is_attn[int(math.log2(res))] = True

        return UNet(
            input_channels=6,
            output_channels=(3 if not learn_sigma else 6),
            n_channels=n_channels,
            ch_mults=channel_mult,
            is_attn=is_attn,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dim_head=dim_head
        ).to(device)

    elif model_name == "MiddleUNet":
        return MiddleUNet(
            vector_conditioning = vector_conditioning,
            input_channels=(6 if not vector_conditioning else 4),
            output_channels=(3 if not learn_sigma else 6),
            n_channels = n_channels,
            image_size = image_size,
            n_blocks = n_blocks
        ).to(device)

    else:
        raise ValueError("Model Type not supported")

def create_diffusion(
        noise_steps,
        image_size,
        device,
        learn_sigma,
        conditioned_prior,
        vector_conditioning
):
    return DiffusionTools(
        noise_steps=noise_steps,
        img_size=image_size,
        conditioned_prior=conditioned_prior,
        vector_conditioning=vector_conditioning,
        learn_sigma=learn_sigma,
        device=device
    )

